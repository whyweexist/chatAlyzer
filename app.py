import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Download NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Set page configuration
st.set_page_config(
    page_title="WhatsApp Chat Analysis",
    page_icon="💬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding: 2rem;
    background-color: #f5f5f5;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
h1, h2, h3 {
    color: #075e54;
}
.metric-card {
    background-color: white;
    border-radius: 5px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title("📱 WhatsApp Chat Analyzer")
st.markdown("""
Upload your WhatsApp chat export (txt file) and get insights about conversation patterns,
user engagement, and predictions on group activity trends.
""")

# Functions for chat processing
def parse_chat(text_data):
    """Extract messages, timestamps, and users from WhatsApp chat text."""
    # Add a new pattern specifically for the format: 13/10/2024, 21:57 - Yuvaan Sir: <Media omitted>
    # pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?) - (.*?): (.*?)(?=\n\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})? - |$)'
    pattern = r'''
    (                               # Start of timestamp
        (?:                         # Date variations:
            \d{1,2}[\/\-\.]         # Day/month separator (1-2 digits)
            (?:\d{1,2}|[A-Za-z]{3}) # Month: 1-2 digits or abbreviated text (handles typos like "D")
            [\/\-\.]                # Year separator
            \d{2,4}                 # Year (2-4 digits)
        )
        ,?\s+                       # Comma and whitespace
        (?:                         # Time variations:
            \d{1,2}:\d{2}(?::\d{2})?      # 24h format (with optional seconds)
            |                               # OR
            \d{1,2}:\d{2}\s*[apAP][mM]     # 12h format with AM/PM
        )
    )                               # End of timestamp
    \s*[-–—]\s*                     # Flexible separator (hyphen, en-dash, em-dash)
    (                               # User (exclude system messages)
        (?!.*?(?:created\sgroup|added\syou|changed\sthe\sgroup|pinned\sa\smessage|Messages\sand\scalls)) 
        .*?                         # Actual user name
    )
    # (.*?)                           # User
    :\s                             # Message separator
    (.*?)                           # Message content
    (?=\n\s*\d|$)                   # Lookahead for next message
    '''
    
    matches = re.findall(pattern, text_data, re.VERBOSE | re.DOTALL | re.IGNORECASE)
    if not matches:
        # Alternative pattern for bracketed timestamps [1/19/88, 21:59]
        alt_pattern = r'\[({0})\]'.format(
            r'(?:\d{1,2}[\/\-\.](?:\d{1,2}|[A-Za-z]+)[\/\-\.]\d{2,4},\s*(?:\d{1,2}:\d{2}(?::\d{2})?|\d{1,2}:\d{2}\s*[apAP][mM])'
        )
        matches = re.findall(alt_pattern + r'\s(.*?):\s(.*?)(?=\n\[|$)', text_data, re.DOTALL)
    
    if not matches:
        st.error("Chat format not recognized. Please check your export.")
        st.code(text_data[:1000], language="text")
        return None
    
    # Extract matches
    # matches = re.findall(pattern, text_data, re.DOTALL)
    
    # Convert to DataFrame
    df = pd.DataFrame(matches, columns=['timestamp', 'user', 'message'])
    
    if len(df) == 0:
        # Try alternative pattern for different timestamp format
        pattern = r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?)\]\s(.*?):\s(.*?)(?=\n\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\]\s|$)'
        matches = re.findall(pattern, text_data, re.DOTALL)
        df = pd.DataFrame(matches, columns=['timestamp', 'user', 'message'])
    
    if len(df) == 0:
        # Another alternative format (no seconds in timestamp)
        pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}) - (.*?): (.*?)(?=\n\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2} - |$)'
        matches = re.findall(pattern, text_data, re.DOTALL)
        df = pd.DataFrame(matches, columns=['timestamp', 'user', 'message'])
    
    if len(df) == 0:
        st.error("Could not parse the chat file. Please check the format and try again.")
        
        # Print sample of the file to help with debugging
        st.code(text_data[:1000], language="text")
        
        # Let's try a more lenient pattern and show what it finds
        simple_pattern = r'(\d{1,2}/\d{1,2}/\d{4}.*?) - (.*?): (.*?)(?=\n\d{1,2}/\d{1,2}/\d{4}|$)'
        sample_matches = re.findall(simple_pattern, text_data[:5000], re.DOTALL)
        if sample_matches:
            st.info(f"Found {len(sample_matches)} messages with a simpler pattern. Here's an example of what was matched:")
            st.write(sample_matches[0])
        
        return None
    
    return df

def process_dataframe(df):
    """Process the dataframe to extract useful features for analysis."""
    if df is None or len(df) == 0:
        return None
    df['timestamp'] = df['timestamp'].str.replace(r'[\[\]]', '', regex=True)
    system_keywords = [
        'created group', 'changed the group', 
        'pinned a message', 'added you',
        'changed this group\'s icon', 'Messages and calls'
    ]
    
    # Use case-insensitive filtering
    df = df[~df['message'].str.contains('|'.join(system_keywords), case=False)]
    # Try different timestamp formats
    # Define all possible datetime formats
    datetime_formats = [
        # Day-first formats
        '%d/%m/%y, %H:%M',        # 19/1/88, 21:59
        '%d/%m/%y, %H:%M:%S',      # 19/1/88, 21:59:59
        '%d/%m/%Y, %H:%M',         # 19/1/1988, 21:59
        '%d-%m-%Y, %H:%M:%S',      # 19-01-1988, 21:59:59
        '%d.%m.%Y, %I:%M %p',      # 19.01.1988, 09:59 PM
        
        # Month-first formats
        '%m/%d/%y, %H:%M',         # 1/19/88, 21:59 (US)
        '%m/%d/%Y, %I:%M %p',      # 1/19/1988, 09:59 AM
        '%m-%d-%y, %H:%M:%S',      # 01-19-88, 21:59:59
        '%m.%d.%Y, %I:%M %p',      # 1.19.1988, 09:59 PM
        
        # Special cases
        '%d/%b/%y, %H:%M',         # Handles month abbreviations (19/Jan/88)
        '%d-%m-%y, %I:%M %p',      # 19-01-88, 09:59 PM
        '%d.%m.%y, %H:%M',         # 19.01.88, 21:59
        '%m/%d/%y, %I:%M%p',       # 1/19/88, 9:59PM (no space)
    ]

    # Try parsing with multiple formats
    for fmt in datetime_formats:
        try:
            df['datetime'] = pd.to_datetime(
                df['timestamp'],
                format=fmt,
                errors='raise',
                dayfirst=True
            )
            break
        except (ValueError, pd.errors.OutOfBoundsDatetime):
            continue
            
    # Final fallback with flexible parser
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(
            df['timestamp'],
            infer_datetime_format=True,
            dayfirst=True,
            errors='coerce'
        )
    # try:
    #     df['datetime'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y, %H:%M')
    # except:
    #     try:
    #         df['datetime'] = pd.to_datetime(df['timestamp'], format='%d/%m/%y, %H:%M')
    #     except:
    #         try:
    #             df['datetime'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y, %I:%M %p')
    #         except:
    #             try:
    #                 df['datetime'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y, %H:%M:%S')
    #             except:
    #                 try:
    #                     df['datetime'] = pd.to_datetime(df['timestamp'], format='%m/%d/%y, %I:%M:%S %p')
    #                 except:
    #                     st.warning("Could not parse datetime format. Using a generic approach.")
    #                     # Generic approach - this might not be accurate
    #                     df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Filter out rows with NaT datetime
    df = df.dropna(subset=['datetime'])
    
    # Extract date components
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['day_of_week'] = df['datetime'].dt.day_name()
    
    # Message features
    df['message_length'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))
    
    # Identify media messages
    df['is_media'] = df['message'].str.contains('<Media omitted>|image omitted|video omitted|audio omitted|document omitted', case=False)
    
    # Identify reactions (thumbs up, heart, etc.) - common WhatsApp reaction patterns
    df['is_reaction'] = df['message'].str.match(r'^(👍|❤️|😂|😮|😢|🙏|👌|👏|🔥|❤|♥️|🤣|😁|👆|👇|👉|👈|🤞|✌️|🤦‍♀️|🤦‍♂️|👨‍💻|👩‍💻|👍🏻|👍🏼|👍🏽|👍🏾|👍🏿)$')
    
    # Identify URL sharing
    url_pattern = r'https?://\S+|www\.\S+'
    df['contains_url'] = df['message'].str.contains(url_pattern, case=False)
    
    # Get sentiment scores
    sid = SentimentIntensityAnalyzer()
    df['sentiment'] = df['message'].apply(lambda x: 
                                         sid.polarity_scores(str(x))['compound'] 
                                         if not pd.isna(x) else 0)
    
    return df
# def parse_chat(text_data):
#     """Extract messages, timestamps, and users from WhatsApp chat text."""
#     # Pattern for messages with date, time, user, and content
#     pattern = r'(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s(?:AM|PM)?) - (.*?): (.*?)(?=\n\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\s(?:AM|PM)? - |$)'
    
#     # Extract matches
#     matches = re.findall(pattern, text_data, re.DOTALL)
    
#     # Convert to DataFrame
#     df = pd.DataFrame(matches, columns=['timestamp', 'user', 'message'])
    
#     if len(df) == 0:
#         # Alternative pattern for different WhatsApp export formats
#         pattern = r'\[(\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?)\]\s(.*?):\s(.*?)(?=\n\[\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}(?::\d{2})?\]\s|$)'
#         matches = re.findall(pattern, text_data, re.DOTALL)
#         df = pd.DataFrame(matches, columns=['timestamp', 'user', 'message'])
    
#     if len(df) == 0:
#         st.error("Could not parse the chat file. Please check the format and try again.")
#         return None
    
#     return df

# def process_dataframe(df):
#     """Process the dataframe to extract useful features for analysis."""
#     if df is None or len(df) == 0:
#         return None
    
#     # Try different timestamp formats
#     try:
#         df['datetime'] = pd.to_datetime(df['timestamp'], format='%m/%d/%y, %I:%M:%S %p')
#     except:
#         try:
#             df['datetime'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y, %H:%M:%S')
#         except:
#             try:
#                 df['datetime'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y, %H:%M')
#             except:
#                 try:
#                     df['datetime'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y, %I:%M %p')
#                 except:
#                     st.warning("Could not parse datetime format. Using a generic approach.")
#                     # Generic approach - this might not be accurate
#                     df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
#     # Filter out rows with NaT datetime
#     df = df.dropna(subset=['datetime'])
    
#     # Extract date components
#     df['date'] = df['datetime'].dt.date
#     df['time'] = df['datetime'].dt.time
#     df['hour'] = df['datetime'].dt.hour
#     df['day'] = df['datetime'].dt.day
#     df['month'] = df['datetime'].dt.month
#     df['year'] = df['datetime'].dt.year
#     df['day_of_week'] = df['datetime'].dt.day_name()
    
#     # Message features
#     df['message_length'] = df['message'].apply(len)
#     df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))
    
#     # Identify media messages
#     df['is_media'] = df['message'].str.contains('<Media omitted>|image omitted|video omitted|audio omitted|document omitted', case=False)
    
#     # Identify reactions (thumbs up, heart, etc.) - common WhatsApp reaction patterns
#     df['is_reaction'] = df['message'].str.match(r'^(👍|❤️|😂|😮|😢|🙏|👌|👏|🔥|❤|♥️|🤣|😁|👆|👇|👉|👈|🤞|✌️|🤦‍♀️|🤦‍♂️|👨‍💻|👩‍💻|👍🏻|👍🏼|👍🏽|👍🏾|👍🏿)$')
    
#     # Identify URL sharing
#     url_pattern = r'https?://\S+|www\.\S+'
#     df['contains_url'] = df['message'].str.contains(url_pattern, case=False)
    
#     # Get sentiment scores
#     sid = SentimentIntensityAnalyzer()
#     df['sentiment'] = df['message'].apply(lambda x: 
#                                          sid.polarity_scores(str(x))['compound'] 
#                                          if not pd.isna(x) else 0)
    
#     return df

def get_user_stats(df):
    """Get per-user statistics."""
    if df is None or len(df) == 0:
        return None
    
    user_stats = df.groupby('user').agg({
        'message': 'count',
        'message_length': 'mean',
        'word_count': ['sum', 'mean'],
        'is_media': 'sum',
        'is_reaction': 'sum',
        'contains_url': 'sum',
        'sentiment': 'mean'
    }).reset_index()
    
    user_stats.columns = ['user', 'message_count', 'avg_message_length', 
                         'total_words', 'avg_words_per_message',
                         'media_shared', 'reactions_sent', 'urls_shared',
                         'avg_sentiment']
    
    # Calculate percentage of total messages
    total_messages = user_stats['message_count'].sum()
    user_stats['message_percentage'] = (user_stats['message_count'] / total_messages * 100).round(2)
    
    return user_stats

def get_time_stats(df):
    """Get time-based statistics."""
    if df is None or len(df) == 0:
        return None
    
    # Messages by date
    daily_messages = df.groupby('date').size().reset_index(name='count')
    daily_messages['date'] = pd.to_datetime(daily_messages['date'])
    daily_messages = daily_messages.sort_values('date')
    
    # Messages by hour
    hourly_messages = df.groupby('hour').size().reset_index(name='count')
    
    # Messages by day of week
    dow_messages = df.groupby('day_of_week').size().reset_index(name='count')
    # Sort by day of week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_messages['day_of_week'] = pd.Categorical(dow_messages['day_of_week'], categories=days_order, ordered=True)
    dow_messages = dow_messages.sort_values('day_of_week')
    
    # Messages by month
    monthly_messages = df.groupby(['year', 'month']).size().reset_index(name='count')
    monthly_messages['year_month'] = monthly_messages.apply(lambda x: f"{int(x['year'])}-{int(x['month']):02d}", axis=1)
    
    return {
        'daily': daily_messages,
        'hourly': hourly_messages,
        'day_of_week': dow_messages,
        'monthly': monthly_messages
    }

def calculate_engagement_metrics(df, time_stats):
    """Calculate metrics for engagement analysis."""
    if df is None or len(df) == 0 or time_stats is None:
        return None
    
 
    # Sort the DataFrame in place
    df.sort_values('datetime', inplace=True)
    
    # Calculate response times
    df['next_message_time'] = df['datetime'].shift(-1)
    df['response_time_seconds'] = (df['next_message_time'] - df['datetime']).dt.total_seconds()
    # Number of active days
    active_days = len(time_stats['daily'])
    
    # Users who sent at least one message
    active_users = df['user'].nunique()
    
    # Average messages per day
    avg_msgs_per_day = len(df) / active_days if active_days > 0 else 0
    
    # Messages per user
    msgs_per_user = len(df) / active_users if active_users > 0 else 0
    
    # Calculate weekly activity
    df['week'] = df['datetime'].dt.isocalendar().week
    df['year_week'] = df['datetime'].dt.strftime('%Y-%U')
    weekly_activity = df.groupby('year_week').size().reset_index(name='count')
    
    # Calculate response times
    df = df.sort_values('datetime')
    df['next_message_time'] = df['datetime'].shift(-1)
    df['response_time_seconds'] = (df['next_message_time'] - df['datetime']).dt.total_seconds()
    # Filter to only include reasonable response times (e.g., < 1 hour)
    reasonable_responses = df[df['response_time_seconds'] < 3600]
    avg_response_time = reasonable_responses['response_time_seconds'].mean()
    
    # Calculate conversation clusters
    # Define a conversation as messages within 30 minutes of each other
    df['prev_message_time'] = df['datetime'].shift(1)
    df['time_since_prev_msg'] = (df['datetime'] - df['prev_message_time']).dt.total_seconds()
    df['new_conversation'] = df['time_since_prev_msg'] > 1800  # 30 minutes in seconds
    df['conversation_id'] = df['new_conversation'].cumsum()
    conversation_counts = df.groupby('conversation_id').size().reset_index(name='messages_in_conversation')
    avg_conversation_length = conversation_counts['messages_in_conversation'].mean()
    
    # Calculate engagement trends over time
    weekly_users = df.groupby('year_week')['user'].nunique().reset_index(name='unique_users')
    weekly_engagement = pd.merge(weekly_activity, weekly_users, on='year_week')
    weekly_engagement['msgs_per_user'] = weekly_engagement['count'] / weekly_engagement['unique_users']
    
    # Calculate engagement scores
    # Simple engagement score: messages per day normalized
    max_daily = time_stats['daily']['count'].max()
    time_stats['daily']['engagement_score'] = time_stats['daily']['count'] / max_daily if max_daily > 0 else 0
    
    # Calculate 7-day rolling average of daily messages
    time_stats['daily']['date'] = pd.to_datetime(time_stats['daily']['date'])
    time_stats['daily'] = time_stats['daily'].sort_values('date')
    time_stats['daily']['rolling_avg'] = time_stats['daily']['count'].rolling(window=7, min_periods=1).mean()
    
    return {
        'active_days': active_days,
        'active_users': active_users,
        'avg_msgs_per_day': avg_msgs_per_day,
        'msgs_per_user': msgs_per_user,
        'weekly_activity': weekly_activity,
        'avg_response_time': avg_response_time,
        'avg_conversation_length': avg_conversation_length,
        'weekly_engagement': weekly_engagement
    }

def predict_future_engagement(df, engagement_metrics):
    """Use ML to predict future engagement trends."""
    if df is None or len(df) == 0 or engagement_metrics is None:
        return None, None
    
    # Create features for prediction
    daily_stats = df.groupby('date').agg({
        'message': 'count',
        'user': 'nunique',
        'is_media': 'sum',
        'is_reaction': 'sum',
        'sentiment': 'mean',
        'contains_url': 'sum'
    }).reset_index()
    
    daily_stats.columns = ['date', 'message_count', 'active_users', 
                         'media_count', 'reaction_count', 'avg_sentiment', 'url_count']
    
    # Generate more features
    daily_stats['date'] = pd.to_datetime(daily_stats['date'])
    daily_stats['day_of_week'] = daily_stats['date'].dt.dayofweek
    daily_stats['is_weekend'] = daily_stats['day_of_week'].isin([5, 6]).astype(int)
    daily_stats['month'] = daily_stats['date'].dt.month
    
    # Add lagged features
    for lag in [1, 3, 7]:
        daily_stats[f'message_count_lag_{lag}'] = daily_stats['message_count'].shift(lag)
        daily_stats[f'active_users_lag_{lag}'] = daily_stats['active_users'].shift(lag)
    
    # Add rolling window features
    for window in [3, 7, 14]:
        daily_stats[f'message_count_roll_{window}'] = daily_stats['message_count'].rolling(window=window).mean()
        daily_stats[f'active_users_roll_{window}'] = daily_stats['active_users'].rolling(window=window).mean()
    
    # Add trend indicators
    daily_stats['message_growth'] = daily_stats['message_count'].pct_change(7)
    daily_stats['user_growth'] = daily_stats['active_users'].pct_change(7)
    
    # Drop NaN values
    daily_stats = daily_stats.dropna()
    
    if len(daily_stats) < 14:  # Need enough data for meaningful prediction
        return None, None
    
    # Define engagement target
    # Using a simplistic approach for demonstration: 
    # days with above-median message count are considered "high engagement"
    median_msgs = daily_stats['message_count'].median()
    daily_stats['high_engagement'] = (daily_stats['message_count'] > median_msgs).astype(int)
    
    # Define features and target
    features = [col for col in daily_stats.columns if col not in 
                ['date', 'high_engagement', 'message_count']]
    
    X = daily_stats[features]
    y = daily_stats['high_engagement']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train random forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get feature importance
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Make future predictions
    # Use the last 14 days to predict next 7 days
    last_day = daily_stats['date'].max()
    
    # Create prediction dataframe with basic features
    future_dates = pd.date_range(start=last_day + pd.Timedelta(days=1), periods=7)
    future_df = pd.DataFrame({'date': future_dates})
    future_df['day_of_week'] = future_df['date'].dt.dayofweek
    future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
    future_df['month'] = future_df['date'].dt.month
    
    # For simplicity, use the averages of the last 14 days for other features
    last_days = daily_stats.tail(14)
    for col in ['active_users', 'media_count', 'reaction_count', 'avg_sentiment', 'url_count']:
        future_df[col] = last_days[col].mean()
    
    # Use the trends from the last available data points
    for lag in [1, 3, 7]:
        future_df[f'message_count_lag_{lag}'] = last_days['message_count'].tail(lag).mean()
        future_df[f'active_users_lag_{lag}'] = last_days['active_users'].tail(lag).mean()
    
    for window in [3, 7, 14]:
        future_df[f'message_count_roll_{window}'] = last_days['message_count'].tail(window).mean()  
        future_df[f'active_users_roll_{window}'] = last_days['active_users'].tail(window).mean()
    
    future_df['message_growth'] = last_days['message_growth'].mean()
    future_df['user_growth'] = last_days['user_growth'].mean()
    
    # Reorder columns to match the training data
    future_df_features = future_df[features]
    
    # Scale and predict
    future_scaled = scaler.transform(future_df_features)
    future_predictions = model.predict(future_scaled)
    future_df['predicted_high_engagement'] = future_predictions
    
    # Calculate probability scores
    future_probs = model.predict_proba(future_scaled)
    future_df['engagement_score'] = future_probs[:, 1]  # Probability of class 1 (high engagement)
    
    return future_df, feature_importance

def generate_wordcloud(df):
    """Generate word cloud from messages."""
    if df is None or len(df) == 0:
        return None
    
    # Combine all messages
    text = ' '.join(df['message'].dropna().astype(str).tolist())
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove common WhatsApp markers
    text = re.sub(r'<Media omitted>|image omitted|video omitted|audio omitted|document omitted', '', text, flags=re.IGNORECASE)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                         colormap='viridis', max_words=100).generate(text)
    
    return wordcloud

def cluster_users(user_stats):
    """Cluster users based on activity patterns."""
    if user_stats is None or len(user_stats) < 3:  # Need at least 3 users for meaningful clustering
        return None
    
    # Select features for clustering
    features = ['message_count', 'avg_message_length', 'avg_words_per_message', 
               'media_shared', 'reactions_sent', 'urls_shared', 'message_percentage']
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(user_stats[features])
    
    # Determine optimal number of clusters - use 3 if enough users, otherwise 2
    n_clusters = min(3, len(user_stats) - 1) if len(user_stats) > 2 else 2
    
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    user_stats['cluster'] = kmeans.fit_predict(X)
    
    # Add descriptive cluster labels based on message count
    cluster_means = user_stats.groupby('cluster')['message_count'].mean()
    cluster_rank = cluster_means.rank(ascending=False)
    
    # Create mapping dictionary
    cluster_names = {}
    for cluster, rank in cluster_rank.items():
        if rank == 1:
            cluster_names[cluster] = "Very Active Users"
        elif rank == 2:
            cluster_names[cluster] = "Moderately Active Users"
        else:
            cluster_names[cluster] = "Less Active Users"
    
    # Apply mapping
    user_stats['user_type'] = user_stats['cluster'].map(cluster_names)
    
    return user_stats

def analyze_engagement_trend(time_stats, engagement_metrics):
    """Determine if engagement is increasing, decreasing, or stable."""
    if time_stats is None or engagement_metrics is None:
        return "Unknown", 0
    
    # Use the rolling average to determine trend
    daily = time_stats['daily']
    
    if len(daily) < 14:  # Need enough data
        return "Not enough data", 0
    
    # Get the last 14 days
    recent = daily.sort_values('date').tail(14)
    
    # Split into two periods and compare
    period1 = recent.head(7)['count'].mean()
    period2 = recent.tail(7)['count'].mean()
    
    change_pct = ((period2 - period1) / period1 * 100) if period1 > 0 else 0
    
    if change_pct > 10:
        trend = "Increasing"
    elif change_pct < -10:
        trend = "Decreasing"
    else:
        trend = "Stable"
    
    return trend, change_pct

# File upload
uploaded_file = st.file_uploader("Upload WhatsApp Chat (.txt file)", type="txt")

if uploaded_file is not None:
    # Read and process file
    text_data = uploaded_file.getvalue().decode("utf-8")
    
    with st.spinner('Processing chat data...'):
        # Parse chat
        df = parse_chat(text_data)
        
        if df is not None and len(df) > 0:
            # Process dataframe
            df = process_dataframe(df)
            
            if df is not None and len(df) > 0:
                # Get statistics
                user_stats = get_user_stats(df)
                time_stats = get_time_stats(df)
                engagement_metrics = calculate_engagement_metrics(df, time_stats)
                future_engagement, feature_importance = predict_future_engagement(df, engagement_metrics)
                wordcloud = generate_wordcloud(df)
                clustered_users = cluster_users(user_stats)
                
                # Determine overall engagement trend
                trend, change_pct = analyze_engagement_trend(time_stats, engagement_metrics)
                
                # Display overview
                st.header("📊 Chat Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Total Messages", f"{len(df):,}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Active Users", f"{engagement_metrics['active_users']:,}")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col3:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Time Period", f"{(df['datetime'].max() - df['datetime'].min()).days + 1} days")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with col4:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.metric("Engagement Trend", trend, f"{change_pct:.1f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Display date range
                st.info(f"Chat data from {df['datetime'].min().strftime('%B %d, %Y')} to {df['datetime'].max().strftime('%B %d, %Y')}")
                
                # Message activity trends
                st.header("📈 Message Activity Trends")
                tab1, tab2, tab3 = st.tabs(["Daily Activity", "Time Patterns", "User Activity"])
                
                with tab1:
                    # Daily message count with trend line
                    daily_fig = px.line(time_stats['daily'], x='date', y='count', 
                                        title='Daily Message Count',
                                        labels={'date': 'Date', 'count': 'Number of Messages'})
                    daily_fig.add_scatter(x=time_stats['daily']['date'], y=time_stats['daily']['rolling_avg'],
                                         mode='lines', name='7-Day Rolling Average',
                                         line=dict(color='red', width=2))
                    st.plotly_chart(daily_fig, use_container_width=True)
                    
                    # Monthly message count
                    monthly_fig = px.bar(time_stats['monthly'], x='year_month', y='count',
                                         title='Monthly Message Count',
                                         labels={'year_month': 'Month', 'count': 'Number of Messages'})
                    st.plotly_chart(monthly_fig, use_container_width=True)
                
                with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Hourly distribution
                        hourly_fig = px.bar(time_stats['hourly'], x='hour', y='count',
                                           title='Messages by Hour of Day',
                                           labels={'hour': 'Hour', 'count': 'Number of Messages'})
                        st.plotly_chart(hourly_fig, use_container_width=True)
                    
                    with col2:
                        # Day of week distribution
                        dow_fig = px.bar(time_stats['day_of_week'], x='day_of_week', y='count',
                                        title='Messages by Day of Week',
                                        labels={'day_of_week': 'Day', 'count': 'Number of Messages'})
                        st.plotly_chart(dow_fig, use_container_width=True)
                    
                    # Response time distribution
                    response_times = df[df['response_time_seconds'] < 3600]['response_time_seconds'] / 60
                    fig = px.histogram(response_times, 
                                      title='Response Time Distribution (minutes)',
                                      labels={'value': 'Response Time (minutes)', 'count': 'Frequency'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    if clustered_users is not None:
                        # User message count
                        top_users = clustered_users.sort_values('message_count', ascending=False).head(10)
                        
                        # User clusters visualization
                        cluster_fig = px.scatter(clustered_users, x='message_count', y='avg_words_per_message',
                                               size='message_percentage', color='user_type',
                                               hover_name='user', 
                                               title='User Activity Clusters',
                                               labels={'message_count': 'Number of Messages', 
                                                      'avg_words_per_message': 'Avg Words per Message'})
                        st.plotly_chart(cluster_fig, use_container_width=True)
                        
                        # Top users bar chart
                        user_fig = px.bar(top_users, x='user', y='message_count', 
                                         color='user_type',
                                         title='Top 10 Most Active Users',
                                         labels={'user': 'User', 'message_count': 'Number of Messages'})
                        st.plotly_chart(user_fig, use_container_width=True)
                    
                # User Analysis
                st.header("👥 User Analysis")
                
                if clustered_users is not None:
                    # Show user clusters
                    st.subheader("User Engagement Groups")
                    cluster_counts = clustered_users['user_type'].value_counts().reset_index()
                    cluster_counts.columns = ['User Group', 'Count']
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Pie chart of user types
                        pie_fig = px.pie(cluster_counts, values='Count', names='User Group',
                                        title='Distribution of User Types')
                        st.plotly_chart(pie_fig, use_container_width=True)
                    
                    with col2:
                        # User types table
                        st.dataframe(clustered_users[['user', 'message_count', 'message_percentage', 
                                                    'avg_message_length', 'user_type']].rename(
                                                    columns={
                                                        'user': 'User',
                                                        'message_count': 'Messages',
                                                        'message_percentage': 'Share (%)',
                                                        'avg_message_length': 'Avg Length',
                                                        'user_type': 'User Type'
                                                    }), use_container_width=True)
                
                # Message content analysis
                st.header("📝 Message Content Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Media sharing stats
                    media_count = df['is_media'].sum()
                    url_count = df['contains_url'].sum()
                    reaction_count = df['is_reaction'].sum()
                    
                    content_data = pd.DataFrame({
                        'Type': ['Text Messages', 'Media Messages', 'URL Shares', 'Reactions'],
                        'Count': [len(df) - media_count - url_count - reaction_count, 
                                 media_count, url_count, reaction_count]
                    })
                    
                    content_fig = px.pie(content_data, values='Count', names='Type',
                                        title='Message Content Distribution')
                    st.plotly_chart(content_fig, use_container_width=True)
                
                with col2:
                    # Sentiment distribution
                    sentiment_fig = px.histogram(df, x='sentiment', 
                                               title='Message Sentiment Distribution',
                                               labels={'sentiment': 'Sentiment Score (-1 to 1)'})
                    st.plotly_chart(sentiment_fig, use_container_width=True)
                
                # Word cloud
                if wordcloud:
                    st.subheader("Common Words in Chat")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                
                # Engagement Prediction
                st.header("🔮 Engagement Prediction & Analysis")
                
                # Overall engagement analysis
                st.subheader("Engagement Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Weekly active users trend
                    if 'weekly_engagement' in engagement_metrics:
                        weekly_fig = px.line(engagement_metrics['weekly_engagement'], x='year_week', y='unique_users',
                                           title='Weekly Active Users',
                                           labels={'year_week': 'Week', 'unique_users': 'Number of Users'})
                        st.plotly_chart(weekly_fig, use_container_width=True)
                
                with col2:
                    # Messages per user trend
                    if 'weekly_engagement' in engagement_metrics:
                        msgs_per_user_fig = px.line(engagement_metrics['weekly_engagement'], x='year_week', y='msgs_per_user',
                                                  title='Weekly Messages per User',
                                                  labels={'year_week': 'Week', 'msgs_per_user': 'Messages per User'})
                        st.plotly_chart(msgs_per_user_fig, use_container_width=True)
                
                # Future engagement prediction
                if future_engagement is not None:
                    st.subheader("Future Engagement Forecast")
                    
                    # Display prediction results
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Line chart of predicted engagement scores
                        future_fig = px.line(future_engagement, x='date', y='engagement_score',
                                           title='7-Day Engagement Score Forecast',
                                           labels={'date': 'Date', 'engagement_score': 'Engagement Score'})
                        
                        # Add threshold line for high engagement
                        future_fig.add_shape(
                            type="line",
                            x0=future_engagement['date'].min(),
                            y0=0.5,
                            x1=future_engagement['date'].max(),
                            y1=0.5,
                            line=dict(color="red", dash="dash")
                        )
                        
                        st.plotly_chart(future_fig, use_container_width=True)
                    
                    with col2:
                        # Feature importance
                        if feature_importance is not None:
                            top_features = feature_importance.head(5)
                            feat_fig = px.bar(top_features, y='feature', x='importance', 
                                             title='Top 5 Factors Affecting Engagement',
                                             labels={'importance': 'Importance', 'feature': 'Factor'},
                                             orientation='h')
                            st.plotly_chart(feat_fig, use_container_width=True)
                    
                    # Show prediction table
                    st.write("Daily Engagement Predictions")
                    forecast_df = future_engagement[['date', 'engagement_score', 'predicted_high_engagement']].copy()
                    forecast_df['date'] = forecast_df['date'].dt.strftime('%Y-%m-%d')
                    forecast_df['engagement_level'] = forecast_df['predicted_high_engagement'].apply(
                        lambda x: "High" if x == 1 else "Low"
                    )
                    forecast_df['engagement_score'] = forecast_df['engagement_score'].apply(
                        lambda x: f"{x:.2f}"
                    )
                    st.dataframe(
                        forecast_df[['date', 'engagement_score', 'engagement_level']].rename(
                            columns={
                                'date': 'Date',
                                'engagement_score': 'Engagement Score',
                                'engagement_level': 'Engagement Level'
                            }
                        ),
                        use_container_width=True
                    )
                
                # Overall engagement conclusion
                st.subheader("Engagement Summary")
                
                # Create engagement metrics summary
                engagement_summary = f"""
                - **Overall Trend**: {trend} ({change_pct:.1f}% change in last 14 days)
                - **Active Users**: {engagement_metrics['active_users']} users have sent messages
                - **Messages per Day**: {engagement_metrics['avg_msgs_per_day']:.1f} messages on average
                - **Messages per User**: {engagement_metrics['msgs_per_user']:.1f} messages per user on average
                - **Response Time**: {engagement_metrics['avg_response_time']/60:.1f} minutes average response time
                - **Conversation Length**: {engagement_metrics['avg_conversation_length']:.1f} messages per conversation on average
                """
                
                st.markdown(engagement_summary)
                
                # Final conclusion about group engagement
                if trend == "Increasing":
                    conclusion = "The group shows **increasing engagement** over time, with growing activity levels."
                elif trend == "Decreasing":
                    conclusion = "The group shows **decreasing engagement** over time, with declining activity levels."
                else:
                    conclusion = "The group shows **stable engagement** over time, with consistent activity levels."
                
                most_active_hour = time_stats['hourly'].sort_values('count', ascending=False).iloc[0]['hour']
                most_active_day = time_stats['day_of_week'].sort_values('count', ascending=False).iloc[0]['day_of_week']
                
                st.markdown(f"""
                ### Final Analysis
                
                {conclusion}
                
                **Key Insights**:
                - Most active time: {most_active_hour}:00 hours
                - Most active day: {most_active_day}
                - Top contributor: {user_stats.iloc[0]['user']} ({user_stats.iloc[0]['message_percentage']}% of messages)
                """)
                
                # Export options
                st.header("📤 Export Options")
                
                # Convert DF to CSV
                csv = df.to_csv(index=False).encode('utf-8')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        "Download Full Data as CSV",
                        csv,
                        "whatsapp_chat_data.csv",
                        "text/csv",
                        key='download-csv'
                    )
                
                with col2:
                    st.download_button(
                        "Download User Statistics as CSV",
                        user_stats.to_csv(index=False).encode('utf-8'),
                        "whatsapp_user_stats.csv",
                        "text/csv",
                        key='download-user-stats'
                    )
            else:
                st.error("Error processing chat data. Please check the format.")
        else:
            st.error("Could not parse chat data. Please make sure you're uploading a valid WhatsApp chat export file.")
else:
    # Show tutorial if no file is uploaded
    st.info("👆 Upload a WhatsApp chat export to begin analysis")
    
    # Instructions
    st.markdown("""
    ### How to export your WhatsApp chat:
    
    1. **Open** the WhatsApp chat/group you want to analyze
    2. **Tap** the three dots in the top right corner (⋮)
    3. Select **More** > **Export chat**
    4. Choose **WITHOUT MEDIA**
    5. **Save** or **Share** the exported file (it will be a .txt file)
    6. **Upload** the file using the file uploader above
    
    ### What this tool analyzes:
    
    - **Message patterns** over time (daily, weekly, monthly)
    - **User engagement** metrics and clustering
    - **Content analysis** including sentiment and word frequency
    - **Engagement prediction** for future group activity
    
    ### Privacy Note:
    
    All processing happens in your browser. Your chat data is not stored on any server.
    """)
    
    # Example images
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://via.placeholder.com/400x300?text=WhatsApp+Export+Example", caption="Example: How to export chat")
    with col2:
        st.image("https://via.placeholder.com/400x300?text=Analysis+Example", caption="Example: Analysis output")

if __name__ == "__main__":
    # This allows the app to be run using `streamlit run app.py`
    pass
