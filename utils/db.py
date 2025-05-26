import pandas as pd
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

load_dotenv()

DB_HOST = os.getenv("DATABASE_HOST")
DB_NAME = os.getenv("DATABASE_NAME")
DB_USER = os.getenv("DATABASE_USER")
DB_PASSWORD = os.getenv("DATABASE_PASSWORD")
DB_PORT = os.getenv("DATABASE_PORT")

def get_data_from_supabase():
    # Create SQLAlchemy engine
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    query = """
        SELECT foods.id AS food_id, foods.name AS food_name, 
               STRING_AGG(ingredients.name, ', ') AS ingredients
        FROM foods
        JOIN food_ingredients ON food_ingredients.food_id = foods.id
        JOIN ingredients ON food_ingredients.ingredient_id = ingredients.id
        GROUP BY foods.id, foods.name
    """
    df = pd.read_sql(query, engine)
    return df

def get_id_to_ingredient():
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    query = "SELECT id, name FROM ingredients"
    df = pd.read_sql(query, engine)
    id_to_ingredient = dict(zip(df['id'], df['name']))
    return id_to_ingredient
