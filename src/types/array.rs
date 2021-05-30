use super::Rdx;

impl<T, const N: usize> Rdx for [T; N]
where
    T: Rdx,
{
    #[inline]
    fn cfg_nbuckets() -> usize {
        T::cfg_nbuckets()
    }

    #[inline]
    fn cfg_nrounds() -> usize {
        T::cfg_nrounds() * N
    }

    #[inline]
    fn get_bucket(&self, round: usize) -> usize {
        let i = round / T::cfg_nrounds();
        let j = round % T::cfg_nrounds();
        self[N - i - 1].get_bucket(j)
    }

    #[inline]
    fn reverse(round: usize, bucket: usize) -> bool {
        let j = round % T::cfg_nrounds();
        T::reverse(j, bucket)
    }
}
