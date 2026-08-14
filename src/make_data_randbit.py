from data_generator import generate_dataset_streamed_randbit

generate_dataset_streamed_randbit(
    "ascon_r3_randbit_train", num_samples=500_000, rounds=3,
    chunk_size=200_000, seed=1,
)
generate_dataset_streamed_randbit(
    "ascon_r3_randbit_val", num_samples=50_000, rounds=3,
    chunk_size=200_000, seed=2,
)