import pandas as pd
import numpy as np
import vaex as vx

nyctaxi = vx.open('s3://vaex/taxi/yellow_taxi_2009_2015_f32.hdf5?anon=true')
print(nyctaxi)

# vaex sample
def sample(self, n=None, frac=None, replace=False, weights=None, random_state=None):
    '''Returns a DataFrame with a random set of rows
    {note_copy}
    Provide either n or frac.
    Example:
    >>> import vaex, numpy as np
    >>> df = vaex.from_arrays(s=np.array(['a', 'b', 'c', 'd']), x=np.arange(1,5))
    >>> df
      #  s      x
      0  a      1
      1  b      2
      2  c      3
      3  d      4
    >>> df.sample(n=2, random_state=42) # 2 random rows, fixed seed
      #  s      x
      0  b      2
      1  d      4
    >>> df.sample(frac=1, random_state=42) # 'shuffling'
      #  s      x
      0  c      3
      1  a      1
      2  d      4
      3  b      2
    >>> df.sample(frac=1, replace=True, random_state=42) # useful for bootstrap (may contain repeated samples)
      #  s      x
      0  d      4
      1  a      1
      2  a      1
      3  d      4
    :param int n: number of samples to take (default 1 if frac is None)
    :param float frac: fractional number of takes to take
    :param bool replace: If true, a row may be drawn multiple times
    :param str or expression weights: (unnormalized) probability that a row can be drawn
    :param int or RandomState: seed or RandomState for reproducability, when None a random seed it chosen
    :return: {return_shallow_copy}
    :rtype: DataFrame
    '''
    self = self.extract()
    if type(random_state) == int or random_state is None:
        random_state = np.random.RandomState(seed=random_state)
    if n is None and frac is None:
        n = 1
    elif frac is not None:
        n = int(round(frac * len(self)))
    weights_values = None
    if weights is not None:
        weights_values = self.evaluate(weights)
        weights_values = weights_values / self.sum(weights)
    indices = random_state.choice(len(self), n, replace=replace, p=weights_values)
    return self.take(indices)


df = pd.read_csv('full_replay_dbg.csv')
print(df)
print(df.iloc[0])
print(df.iloc[0].action)
print( type(df.iloc[0].action) )

