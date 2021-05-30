use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::Rdx;

pub(super) type RcInner<T> = Rc<RefCell<NodeInner<T>>>;
pub(super) type RcPruned<T> = Rc<RefCell<NodePruned<T>>>;

#[derive(Clone)]
pub(super) enum Node<T>
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
    pub(super) fn print(&self, depth: usize)
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
                let c: Self = (&(borrowed.child)).into();
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
pub(super) enum NodeLimited<T>
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
    fn from(obj: &'a NodeLimited<T>) -> Self {
        match obj {
            NodeLimited::Inner(inner) => Self::Inner(inner.clone()),
            NodeLimited::Child(x) => Self::Child(x.clone()),
        }
    }
}

#[derive(Clone)]
pub(super) struct NodeInner<T>
where
    T: Clone + Rdx,
{
    round: usize,
    pub(super) children: Vec<Node<T>>,
}

#[derive(Clone)]
pub(super) struct NodePruned<T>
where
    T: Clone + Rdx,
{
    round: usize,
    nbuckets: usize,
    pub(super) buckets: Vec<usize>,
    pub(super) child: NodeLimited<T>,
}

impl<T> NodeInner<T>
where
    T: Clone + Rdx,
{
    pub(super) fn new(round: usize, nbuckets: usize) -> Self {
        let mut children = Vec::with_capacity(nbuckets);
        for _ in 0..nbuckets {
            children.push(Node::Free);
        }
        Self { round, children }
    }

    pub(super) fn insert(&mut self, x: T) {
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
                    Node::Inner(_) | Node::Pruned(_) => unreachable!(),
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

    pub(super) fn nnodes(&self) -> (usize, usize, usize, usize) {
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
    fn new(round: usize, nbuckets: usize, x: T) -> Self {
        let mut buckets = Vec::with_capacity(round);
        for i in 0..round {
            let r = round - i;
            let bucket = x.get_bucket(r - 1);
            buckets.push(bucket);
        }

        let child = NodeLimited::Child(x);
        Self {
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
                    let tail = Rc::new(RefCell::new(Self {
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
                    let head = Rc::new(RefCell::new(Self {
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
