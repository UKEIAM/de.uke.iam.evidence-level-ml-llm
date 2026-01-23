from sentence_transformers import SentenceTransformer


def prepare_embeddings(train_split_df, test_split_df, runs_config):
    """Generates embeddings for training and testing X data using SentenceTransformer."""
    # Initialize SentenceTransformer model for embeddings
    embedding_model = SentenceTransformer(runs_config["embeddings"]["model"])
    # Generate embeddings for training and testing data
    X_train = embedding_model.encode(
        train_split_df["text"].tolist(),
        show_progress_bar=True,
        max_length=runs_config["embeddings"]["max_length"],
        batch_size=runs_config["embeddings"]["batch_size"],
    )
    X_test = embedding_model.encode(
        test_split_df["text"].tolist(),
        show_progress_bar=True,
        max_length=runs_config["embeddings"]["max_length"],
        batch_size=runs_config["embeddings"]["batch_size"],
    )
    return X_train, X_test
