"""hypothesis strategies to sample data."""

from hypothesis import strategies as st

number_of_data = st.integers(min_value=1, max_value=10000)
batch_size = st.sampled_from([4, 16, 64, 256])
sequence_length = st.integers(min_value=1, max_value=200)
number_of_features = st.integers(min_value=1, max_value=50)
