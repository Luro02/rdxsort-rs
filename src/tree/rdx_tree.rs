use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

use super::node::{Node, NodeInner, NodeLimited, RcInner};

use crate::Rdx;

pub struct RdxTree<T>
where
    T: Clone + Rdx,
{
    root: Node<T>,
}

impl<T> RdxTree<T>
where
    T: Clone + Rdx,
{
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, x: T) {
        match &mut self.root {
            Node::Inner(inner) => {
                inner.borrow_mut().insert(x);
            }
            _ => unreachable!(),
        }
    }

    pub fn iter<'a>(&self) -> RdxTreeIter<'a, T> {
        let mut stack = Vec::new();

        match &self.root {
            Node::Inner(inner) => {
                stack.push((inner.clone(), 1, false));
            }
            _ => unreachable!(),
        }
        RdxTreeIter {
            stack,
            phantom: PhantomData,
        }
    }

    pub fn nnodes(&self) -> (usize, usize, usize, usize) {
        if let Node::Inner(inner) = &self.root {
            inner.borrow().nnodes()
        } else {
            unreachable!()
        }
    }
}

impl<T> fmt::Display for RdxTree<T>
where
    T: Clone + Rdx + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.root.fmt(f)
    }
}

impl<T> Default for RdxTree<T>
where
    T: Clone + Rdx,
{
    fn default() -> Self {
        let rounds = <T as Rdx>::cfg_nrounds();
        let buckets = <T as Rdx>::cfg_nbuckets();
        Self {
            root: Node::Inner(Rc::new(RefCell::new(NodeInner::<T>::new(rounds, buckets)))),
        }
    }
}

pub struct RdxTreeIter<'a, T>
where
    T: Clone + Rdx + 'a,
{
    // iterator stack:
    //   - reference to inner node
    //     (do not work with iterators directly since we need a checked but dynamic borrow)
    //   - current iterator state + 1 (so `0` encodes the "the one BEFORE beginning)
    //   - reverse the iterator order for this subpart if `True`
    stack: Vec<(RcInner<T>, usize, bool)>,

    // keep tree borrow intact
    phantom: PhantomData<&'a RdxTree<T>>,
}

impl<'a, T> Iterator for RdxTreeIter<'a, T>
where
    T: Clone + Rdx + 'a,
{
    type Item = T; // XXX: do not copy!

    fn next(&mut self) -> Option<Self::Item> {
        // the iteration is basically the processing of a stack machine

        let mut result: Option<T> = None;

        // iterate until stack is empty or we have a result
        while !self.stack.is_empty() && result.is_none() {
            // the stack is immutable since we work with the current state,
            // therefore we need to store pending operations (push or pop) and execute afterwards
            let mut push: Option<(Rc<RefCell<NodeInner<T>>>, bool)> = None;
            let mut pop = false;
            let stacksize = self.stack.len();

            if let Some(state) = self.stack.last_mut() {
                let &mut (ref rc, ref mut i, reverse) = state;
                let borrowed = rc.borrow();

                // bounds check for current iterator state
                if (reverse && (*i == 0)) || (*i > borrowed.children.len()) {
                    pop = true;
                } else {
                    // bounds are fine => inspect current sub-element
                    match &borrowed.children[*i - 1] {
                        Node::Free => {
                            // it's a free node, we can ignore that and continue with the iteration
                        }
                        Node::Child(x) => {
                            // we have found some usable data :)
                            result = Some(x.clone());
                        }
                        Node::Inner(inner) => {
                            // inner node => push a new state to the stack
                            let round = <T as Rdx>::cfg_nrounds() - stacksize;
                            let rev = reverse ^ <T as Rdx>::reverse(round, *i - 1);
                            push = Some((inner.clone(), rev));
                        }
                        Node::Pruned(pruned) => {
                            // pruned tree part => let's check what the child is
                            let borrowed2 = pruned.borrow();
                            match &borrowed2.child {
                                NodeLimited::Child(x) => {
                                    // usable data :)
                                    result = Some(x.clone());
                                }
                                NodeLimited::Inner(inner) => {
                                    // simulate traversal of pruned tree part to recover `reverse`
                                    let mut round = <T as Rdx>::cfg_nrounds() - stacksize;
                                    let mut rev = reverse ^ <T as Rdx>::reverse(round, *i - 1);
                                    for j in &borrowed2.buckets {
                                        round += 1;
                                        rev ^= <T as Rdx>::reverse(round, *j);
                                    }

                                    push = Some((inner.clone(), rev));
                                }
                            }
                        }
                    }

                    if reverse {
                        *i -= 1;
                    } else {
                        *i += 1;
                    }
                }
            } else {
                // that cannot happen since we have already checked if the stack is not empty
                unreachable!();
            }

            // execute all pending operations
            if pop {
                self.stack.pop();
            } else if let Some((next, rev)) = push {
                // the iteration of the next stack part starts either at the beginning or end,
                // depending on the fact that it is a reversed iteration or not
                let idx_start = if rev { next.borrow().children.len() } else { 1 };
                self.stack.push((next, idx_start, rev));
            }
        }

        // this can be `None` here in case we've finished iteration and the stack is empty
        result
    }
}
