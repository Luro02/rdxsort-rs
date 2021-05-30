use super::Rdx;

use std::cell::RefCell;
use std::fmt;
use std::marker::PhantomData;
use std::rc::Rc;

type RcInner<T> = Rc<RefCell<NodeInner<T>>>;
type RcPruned<T> = Rc<RefCell<NodePruned<T>>>;

#[derive(Clone)]
enum Node<T>
where
    T: Clone + Rdx,
{
    Inner(RcInner<T>),
    Pruned(RcPruned<T>),
    Child(T),
    Free,
}

impl<T> Node<T>
where
    T: Clone + Rdx,
{
    fn print(&self, depth: usize)
    where
        T: fmt::Display,
    {
        let prefix: String = (0..depth).map(|_| ' ').collect();
        match self {
            Node::Inner(inner) => {
                for (i, c) in inner.borrow().children.iter().enumerate() {
                    println!("{}{}:", prefix, i);
                    c.print(depth + 1);
                }
            }
            Node::Pruned(pruned) => {
                let borrowed = pruned.borrow();
                println!("{}P: [{:?}]", prefix, borrowed.buckets);
                let c: Node<T> = (&(borrowed.child)).into();
                c.print(depth + borrowed.buckets.len());
            }
            Node::Child(x) => {
                println!("{}=> {}", prefix, x);
            }
            Node::Free => {
                println!("{}X", prefix);
            }
        }
    }
}

#[derive(Clone)]
enum NodeLimited<T>
where
    T: Clone + Rdx,
{
    Inner(RcInner<T>),
    Child(T),
}

impl<'a, T> From<&'a NodeLimited<T>> for Node<T>
where
    T: Clone + Rdx,
{
    fn from(obj: &'a NodeLimited<T>) -> Node<T> {
        match obj {
            NodeLimited::Inner(inner) => Node::Inner(inner.clone()),
            NodeLimited::Child(x) => Node::Child(x.clone()),
        }
    }
}

#[derive(Clone)]
struct NodeInner<T>
where
    T: Clone + Rdx,
{
    round: usize,
    children: Vec<Node<T>>,
}

#[derive(Clone)]
struct NodePruned<T>
where
    T: Clone + Rdx,
{
    round: usize,
    nbuckets: usize,
    buckets: Vec<usize>,
    child: NodeLimited<T>,
}

impl<T> NodeInner<T>
where
    T: Clone + Rdx,
{
    fn new(round: usize, nbuckets: usize) -> NodeInner<T> {
        let mut children = Vec::with_capacity(nbuckets);
        for _ in 0..nbuckets {
            children.push(Node::Free);
        }
        NodeInner { round, children }
    }

    fn insert(&mut self, x: T) {
        let bucket = x.get_bucket(self.round - 1);

        if self.round > 1 {
            let clen = self.children.len();
            let replace = {
                match &mut self.children[bucket] {
                    Node::Free => {
                        let pruned =
                            Rc::new(RefCell::new(NodePruned::new(self.round - 1, clen, x)));
                        Some(Node::Pruned(pruned))
                    }
                    Node::Inner(inner) => {
                        inner.borrow_mut().insert(x);
                        None
                    }
                    Node::Pruned(pruned) => Some(pruned.borrow().insert_or_split(x)),
                    Node::Child(_) => unreachable!(),
                }
            };

            if let Some(obj) = replace {
                self.children[bucket] = obj;
            }
        } else {
            let alloc = {
                match self.children[bucket] {
                    Node::Free => true,
                    Node::Child(_) => false,
                    Node::Inner(_) => unreachable!(),
                    Node::Pruned(_) => unreachable!(),
                }
            };

            if alloc {
                self.children[bucket] = Node::Child(x);
            } else if let Node::Child(y) = &mut self.children[bucket] {
                *y = x; // XXX: is that a good idea?
            } else {
                unreachable!()
            }
        }
    }

    fn nnodes(&self) -> (usize, usize, usize, usize) {
        let mut result = (1, 0, 0, 0);
        for c in &self.children {
            match c {
                Node::Inner(inner) => {
                    let tmp = inner.borrow().nnodes();
                    result.0 += tmp.0;
                    result.1 += tmp.1;
                    result.2 += tmp.2;
                    result.3 += tmp.3;
                }
                Node::Pruned(pruned) => {
                    let tmp = pruned.borrow().nnodes();
                    result.0 += tmp.0;
                    result.1 += tmp.1;
                    result.2 += tmp.2;
                    result.3 += tmp.3;
                }
                Node::Child(_) => {
                    result.2 += 1;
                }
                Node::Free => {
                    result.3 += 1;
                }
            }
        }
        result
    }
}

impl<T> NodePruned<T>
where
    T: Clone + Rdx,
{
    fn new(round: usize, nbuckets: usize, x: T) -> NodePruned<T> {
        let mut buckets = Vec::with_capacity(round);
        for i in 0..round {
            let r = round - i;
            let bucket = x.get_bucket(r - 1);
            buckets.push(bucket);
        }

        let child = NodeLimited::Child(x);
        NodePruned {
            round,
            nbuckets,
            buckets,
            child,
        }
    }

    fn insert_or_split(&self, x: T) -> Node<T> {
        for i in 0..self.buckets.len() {
            let r = self.round - i;
            let bucket_y = self.buckets[i];
            let bucket_x = x.get_bucket(r - 1);

            if bucket_x != bucket_y {
                // === outcome a: split ===
                //
                //     [head][middle/diff][tail]
                //
                // becomes
                //
                //                     |-[tail1]
                //     [head]-[middle]-|
                //                     |-[tail2]
                //

                // split head, middle and tail
                let mut buckets_head = self.buckets.clone();
                let buckets_tail = buckets_head.split_off(i + 1);
                buckets_head.pop(); // remove middle part

                // inner node = middle part
                let mut inner = NodeInner::new(self.round - buckets_head.len(), self.nbuckets);

                // add old tail and new branch to inner node
                if buckets_tail.is_empty() {
                    inner.children[bucket_y] = (&self.child).into();
                } else {
                    let tail = Rc::new(RefCell::new(NodePruned {
                        round: self.round - i - 1,
                        nbuckets: self.nbuckets,
                        buckets: buckets_tail,
                        child: self.child.clone(),
                    }));
                    inner.children[bucket_y] = Node::Pruned(tail);
                }
                inner.insert(x);

                // either return inner node (when head is empty) or create new head
                if buckets_head.is_empty() {
                    return Node::Inner(Rc::new(RefCell::new(inner)));
                } else {
                    let head = Rc::new(RefCell::new(NodePruned {
                        round: self.round,
                        nbuckets: self.nbuckets,
                        buckets: buckets_head,
                        child: NodeLimited::Inner(Rc::new(RefCell::new(inner))),
                    }));
                    return Node::Pruned(head);
                }
            }
        }

        // === outcome b: insert ===
        // INFO: Copying seems to be faster than returning an Option and do the change in-place.
        //       I don't know why this is the case.
        let mut cpy = self.clone();
        match &mut cpy.child {
            NodeLimited::Inner(inner) => {
                inner.borrow_mut().insert(x);
            }
            NodeLimited::Child(y) => {
                *y = x;
            }
        }
        Node::Pruned(Rc::new(RefCell::new(cpy)))
    }

    fn nnodes(&self) -> (usize, usize, usize, usize) {
        let mut result = (0, 1, 0, 0);
        match &self.child {
            NodeLimited::Inner(inner) => {
                let tmp = inner.borrow().nnodes();
                result.0 += tmp.0;
                result.1 += tmp.1;
                result.2 += tmp.2;
                result.3 += tmp.3;
            }
            NodeLimited::Child(_) => {
                result.2 += 1;
            }
        }
        result
    }
}

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
            _ => {
                unreachable!();
            }
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

    pub fn print(&self)
    where
        T: fmt::Display,
    {
        self.root.print(0);
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
