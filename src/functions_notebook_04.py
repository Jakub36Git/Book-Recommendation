import pandas as pd
import re

def split_custom_segment(
    ratings_df,
    author_name=None,         # str or list of str
    title_keyword=None,       # str or list of str
    year_range=None,          # (start_year, end_year)
    min_rating=None,
    max_rating=None,
    age_group=None            # e.g. 'teen', 'adult'
):
    """
    Splits ratings_df into a target segment and the rest, based on flexible filtering:
    - Author name(s)
    - Title keyword(s)
    - Year of publication range
    - Rating thresholds (min/max)
    - Age group

    Parameters:
        ratings_df (pd.DataFrame): Cleaned ratings dataset
        author_name (str or list): Author(s) to include (partial match)
        title_keyword (str or list): Book title keywords to include (partial match)
        year_range (tuple): (start_year, end_year)
        min_rating (int): Lower bound on Book-Rating
        max_rating (int): Upper bound on Book-Rating
        age_group (str): Exact Age_Group string to match
    
    Returns:
        target_df (pd.DataFrame): Filtered segment
        rest_df (pd.DataFrame): All other rows
        target_user_ids (set): Unique User-IDs in target_df
    """

    # Start with all True mask
    condition = pd.Series([True] * len(ratings_df), index=ratings_df.index)

    # Filter by author(s)
    if author_name:
        if isinstance(author_name, list):
            author_pattern = '|'.join([re.escape(a) for a in author_name])
            author_condition = ratings_df['Author-Surename'].str.contains(author_pattern, case=False, na=False, regex=True)
        else:
            author_condition = ratings_df['Author-Surename'].str.contains(author_name, case=False, na=False)
        condition &= author_condition

    # Filter by title keyword(s)
    if title_keyword:
        if isinstance(title_keyword, list):
            title_pattern = '|'.join([re.escape(t) for t in title_keyword])
            title_condition = ratings_df['Book-Title'].str.contains(title_pattern, case=False, na=False, regex=True)
        else:
            title_condition = ratings_df['Book-Title'].str.contains(title_keyword, case=False, na=False)
        condition &= title_condition

    # Filter by year range
    if year_range:
        start_year, end_year = year_range
        years = pd.to_numeric(ratings_df['Year-Of-Publication'], errors='coerce')
        condition &= (years >= start_year) & (years <= end_year)

    # Filter by rating thresholds
    if min_rating is not None:
        condition &= ratings_df['Book-Rating'] >= min_rating
    if max_rating is not None:
        condition &= ratings_df['Book-Rating'] <= max_rating

    # === Filter by age group ===
    if age_group:
        if isinstance(age_group, list):
            condition &= ratings_df['Age_Group'].isin(age_group)
        else:
            condition &= ratings_df['Age_Group'] == age_group

    # Final split
    target_user_ids = set(ratings_df[condition]['User-ID'].unique())
    target_df = ratings_df[ratings_df['User-ID'].isin(target_user_ids)].copy()
    rest_df = ratings_df[~ratings_df['User-ID'].isin(target_user_ids)].copy()

    return target_df, rest_df, target_user_ids


def compute_lift(target_df, full_df, min_target_support=20, mode="read"):
    """
    Computes lift scores for books based on either:
    - 'read': count of unique readers
    - 'rating': proportion of positive ratings (>7)

    Parameters:
        target_df (pd.DataFrame): Target segment (must include 'User-ID', 'ISBN', 'Book-Rating')
        full_df (pd.DataFrame): Full dataset
        min_target_support (int): Minimum support (readers or positive ratings)
        mode (str): 'read' or 'rating'

    Returns:
        pd.DataFrame with Lift and supporting metadata
    """
    if mode == "read":
        users_segment = target_df['User-ID'].nunique()
        users_global = full_df['User-ID'].nunique()

        # Count readers per ISBN
        segment_counts = target_df.groupby('ISBN')['User-ID'].nunique().rename('Segment_Readers')
        global_counts = full_df.groupby('ISBN')['User-ID'].nunique().rename('Global_Readers')

        lift_df = pd.concat([segment_counts, global_counts], axis=1).dropna()
        lift_df['P_segment'] = lift_df['Segment_Readers'] / users_segment
        lift_df['P_global'] = lift_df['Global_Readers'] / users_global
        lift_df['Lift'] = lift_df['P_segment'] / lift_df['P_global']
        lift_df = lift_df[lift_df['Segment_Readers'] >= min_target_support]

    elif mode == "rating":
        # Filter to only positive ratings
        segment_pos = target_df[target_df['Book-Rating'] > 7]
        global_pos = full_df[full_df['Book-Rating'] > 7]

        users_segment = target_df['User-ID'].nunique()
        users_global = full_df['User-ID'].nunique()

        # Count positive ratings per book
        segment_counts = segment_pos.groupby('ISBN')['User-ID'].nunique().rename('Segment_Positive')
        global_counts = global_pos.groupby('ISBN')['User-ID'].nunique().rename('Global_Positive')

        lift_df = pd.concat([segment_counts, global_counts], axis=1).dropna()
        lift_df['P_segment'] = lift_df['Segment_Positive'] / users_segment
        lift_df['P_global'] = lift_df['Global_Positive'] / users_global
        lift_df['Lift'] = lift_df['P_segment'] / lift_df['P_global']
        lift_df = lift_df[lift_df['Segment_Positive'] >= min_target_support]

    else:
        raise ValueError("Invalid mode: choose 'read' or 'rating'")

    # Add metadata
    book_info = full_df.drop_duplicates('ISBN').set_index('ISBN')[['Book-Title', 'Author-Surename']]
    lift_df = lift_df.join(book_info)

    # Column ordering
    if mode == "read":
        lift_df = lift_df[['Book-Title', 'Author-Surename', 'Segment_Readers', 'Global_Readers', 'Lift']]
    else:
        lift_df = lift_df[['Book-Title', 'Author-Surename', 'Segment_Positive', 'Global_Positive', 'Lift']]

    return lift_df.sort_values('Lift', ascending=False)

def display_recommendations_knn(recommendations_df, author_name, title_keyword=None):
    """
    Pretty display of recommendation results:
    - Up to 5 top books by the target author
    - Up to 10 top books from other authors (one per author)
    - Uses Book-Author (not just surname)
    - Excludes books that contain the title_keyword (if provided)
    """

    # Normalize author name input
    if isinstance(author_name, list):
        author_label = ", ".join([a.title() for a in author_name])
        author_name_lower = [a.lower() for a in author_name]
    else:
        author_label = author_name.title()
        author_name_lower = [author_name.lower()]

    # Filter books by target author
    is_target_author = recommendations_df['Author-Surename'].str.lower().isin(author_name_lower)
    target_author_books = recommendations_df[is_target_author].copy()
    other_author_books = recommendations_df[~is_target_author].copy()

    # --- Exclude books matching title_keyword ---
    if title_keyword:
        if isinstance(title_keyword, list):
            for kw in title_keyword:
                target_author_books = target_author_books[
                    ~target_author_books['Book-Title'].str.lower().str.contains(kw.lower(), na=False)
                ]
                other_author_books = other_author_books[
                    ~other_author_books['Book-Title'].str.lower().str.contains(kw.lower(), na=False)
                ]
        else:
            target_author_books = target_author_books[
                ~target_author_books['Book-Title'].str.lower().str.contains(title_keyword.lower(), na=False)
            ]
            other_author_books = other_author_books[
                ~other_author_books['Book-Title'].str.lower().str.contains(title_keyword.lower(), na=False)
            ]

    # Take top 5 (or fewer) from target author
    top_target_books = target_author_books.sort_values(by='Similarity', ascending=False).head(5)

    # From other authors, take top 1 book per surname, limit to 10
    top_other_books = (
        other_author_books
        .sort_values(by='Similarity', ascending=False)
        .drop_duplicates(subset='Author-Surename')
        .head(10)
    )

    # Print summary
    print("\n📚 Recommendation Summary:")

    if title_keyword:
        if isinstance(title_keyword, list):
            keyword_label = ", ".join(title_keyword)
        else:
            keyword_label = title_keyword
        print(f"If you just read **{keyword_label}** by **{author_label}**, you might also enjoy:\n")
    else:
        print(f"If you just read books by **{author_label}**, you might also enjoy:\n")

    if not top_target_books.empty:
        print(f"🔁 More from {author_label}:")
        for _, row in top_target_books.iterrows():
            print(f"- {row['Book-Title']} (score: {row['Similarity']:.3f})")
        print()

    if not top_other_books.empty:
        print("🎯 Books from other authors:")
        for _, row in top_other_books.iterrows():
            print(f"- {row['Book-Author']} — {row['Book-Title']} (score: {row['Similarity']:.3f})")