import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity


def compute_knn_similarity(target_df, full_df, min_readers=20, top_n=20, mode="read"):
    """
    Recommends books based on similarity between the target group's behavior and all books.

    Parameters:
        target_df (pd.DataFrame): Ratings for the target segment
        full_df (pd.DataFrame): Full ratings dataset
        min_readers (int): Minimum number of users a book must have to be considered
        top_n (int): Number of recommendations to return
        mode (str): 'read' for binary matrix, 'rating' for Book-Rating values

    Returns:
        pd.DataFrame: Top N recommended books with similarity scores and metadata
    """
    full_df = full_df.copy()

    # Filter books to those present in target segment
    target_isbns = target_df['ISBN'].unique()
    filtered_df = full_df[full_df['ISBN'].isin(target_isbns)]

    # Filter books by popularity
    book_counts = filtered_df['ISBN'].value_counts()
    popular_isbns = book_counts[book_counts >= min_readers].index
    filtered_df = filtered_df[filtered_df['ISBN'].isin(popular_isbns)]

    # Select value column based on mode
    if mode == "read":
        filtered_df["Read"] = 1
        value_column = "Read"
    elif mode == "rating":
        value_column = "Book-Rating"
    else:
        raise ValueError("Invalid mode: choose 'read' or 'rating'")

    # Create book-user matrix
    book_user_matrix = filtered_df.pivot_table(
        index="ISBN",
        columns="User-ID",
        values=value_column,
        fill_value=0
    )

    # --- Create target vector ---
    if mode == "read":
        target_users = [uid for uid in target_df['User-ID'].unique() if uid in book_user_matrix.columns]
        if not target_users:
            raise ValueError("No target users found in book-user matrix.")

        aligned_target_vector = pd.Series(0, index=book_user_matrix.columns)
        aligned_target_vector.loc[target_users] = 1
        target_vector = aligned_target_vector.values.reshape(1, -1)

    else:  # mode == "rating"
        # Mean rating by each user in the target segment
        user_means = target_df.groupby('User-ID')['Book-Rating'].mean()
        aligned_target_vector = book_user_matrix.columns.to_series().map(user_means).fillna(0)
        target_vector = aligned_target_vector.values.reshape(1, -1)

    # --- Compute cosine similarity ---
    similarity_scores = cosine_similarity(book_user_matrix.values, target_vector).flatten()

    # Build result DataFrame
    similarity_df = pd.DataFrame({
        "ISBN": book_user_matrix.index,
        "Similarity": similarity_scores
    })

   # Add metadata
    book_info = full_df.drop_duplicates('ISBN').set_index('ISBN')[['Book-Title', 'Book-Author', 'Author-Surename']]
    similarity_df = similarity_df.set_index('ISBN').join(book_info)
    
    # Return top N
    similarity_df = similarity_df.reset_index().sort_values(by='Similarity', ascending=False)
    return similarity_df.head(top_n)



def compute_knn_author_similarity(target_df, full_df, min_readers=20, top_n=10, mode="read"):
    """
    Recommends authors based on similarity between the target group's behavior and all other authors.

    Parameters:
        target_df (pd.DataFrame): Ratings for the target segment
        full_df (pd.DataFrame): Full ratings dataset
        min_readers (int): Minimum number of users an author must have to be considered
        top_n (int): Number of recommendations to return
        mode (str): 'read' or 'rating'

    Returns:
        pd.DataFrame: Top N recommended authors with similarity scores
    """
    full_df = full_df.copy()

    # Ensure 'Read' column if needed
    if mode == "read":
        full_df["Read"] = 1
        value_column = "Read"
    elif mode == "rating":
        value_column = "Book-Rating"
    else:
        raise ValueError("Invalid mode: choose 'read' or 'rating'")

    # Filter unpopular authors
    author_user_counts = full_df.groupby('Author-Surename')['User-ID'].nunique()
    popular_authors = author_user_counts[author_user_counts >= min_readers].index
    filtered_df = full_df[full_df['Author-Surename'].isin(popular_authors)]

    # Author-user matrix: mean rating or binary read behavior
    author_user_matrix = filtered_df.pivot_table(
        index='Author-Surename',
        columns='User-ID',
        values=value_column,
        aggfunc='mean',
        fill_value=0
    )

    # Target vector (mean rating per user from target_df)
    target_users = [uid for uid in target_df['User-ID'].unique() if uid in author_user_matrix.columns]

    if not target_users:
        raise ValueError("No target users found in author-user matrix.")

    if mode == "read":
        aligned_vector = pd.Series(0, index=author_user_matrix.columns)
        aligned_vector.loc[target_users] = 1
    else:  # rating mode
        user_means = target_df.groupby('User-ID')['Book-Rating'].mean()
        aligned_vector = author_user_matrix.columns.to_series().map(user_means).fillna(0)

    target_vector = aligned_vector.values.reshape(1, -1)

    # Compute similarity
    similarity_scores = cosine_similarity(author_user_matrix.values, target_vector).flatten()

    # Return top N authors
    similarity_df = pd.DataFrame({
        "Author-Surename": author_user_matrix.index,
        "Similarity": similarity_scores
    }).sort_values(by='Similarity', ascending=False).head(top_n)

    return similarity_df
