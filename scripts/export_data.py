from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get credentials from the environment
DB_HOST = os.getenv("my_host")
DB_PORT = os.getenv("my_port")
DB_NAME = os.getenv("my_name")
DB_USER = os.getenv("my_user")
DB_PASSWORD = os.getenv("my_password")
# Create connection to MySQL
def export_data_to_mysql(df):
    engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    # Export DataFrame to MySQL
    df.to_sql('user_satisfaction', con=engine, index=False, if_exists='replace')

    print("Data exported successfully to MySQL!")
