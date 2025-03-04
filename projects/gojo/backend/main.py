from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import shutil
import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.environ["OPENAI_API_KEY"] = os.getenv('OAK')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class CSVAgent:
    def __init__(self, csv_path: str = None, csv_data: pd.DataFrame = None):
        """Initialize with either a path to a CSV file or a pandas DataFrame."""
        if csv_data is not None:
            self.df = csv_data
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Either csv_path or csv_data must be provided")
        
        self.csv_path = csv_path
        self.column_info = self._get_column_info()
        
        # Initialize LLM using OpenAI model (e.g., gpt-3.5-turbo)
        self.llm = OpenAI(model="gpt-3.5-turbo")
        
        # Create tools
        self.tools = [
            FunctionTool.from_defaults(fn=self.get_dataframe_info),
            FunctionTool.from_defaults(fn=self.get_column_data),
            FunctionTool.from_defaults(fn=self.query_data),
            FunctionTool.from_defaults(fn=self.filter_data),
            FunctionTool.from_defaults(fn=self.group_by_data),
            FunctionTool.from_defaults(fn=self.sort_data),
            FunctionTool.from_defaults(fn=self.get_statistics)
        ]
        
        # Create agent
        self.agent = OpenAIAgent.from_tools(
            self.tools,
            llm=self.llm,
            verbose=True,
            system_prompt=(
                "You are a helpful assistant that analyzes CSV data. "
                "Use the available tools to explore, analyze, and visualize the data. "
                "When using tools that return data, summarize the results in a clear and helpful way. "
                "For numerical data, consider providing statistical insights. "
                "For categorical data, consider providing distributions or patterns."
            )
        )
    
    def _get_column_info(self) -> Dict[str, Any]:
        """Get information about the columns in the DataFrame."""
        info = {}
        for col in self.df.columns:
            info[col] = {
                "dtype": str(self.df[col].dtype),
                "sample_values": self.df[col].head(3).tolist(),
                "unique_count": self.df[col].nunique(),
                "null_count": self.df[col].isna().sum()
            }
        return info
    
    def get_dataframe_info(self) -> str:
        """Get basic information about the dataframe."""
        shape = self.df.shape
        columns = self.df.columns.tolist()
        dtypes = self.df.dtypes.to_dict()
        dtypes = {k: str(v) for k, v in dtypes.items()}
        
        missing_data = self.df.isna().sum().to_dict()
        
        info = {
            "rows": shape[0],
            "columns": shape[1],
            "column_names": columns,
            "dtypes": dtypes,
            "missing_values": missing_data,
            "sample_data": self.df.head(5).to_dict(orient="records")
        }
        
        return str(info)
    
    def get_column_data(self, column_name: str) -> str:
        """Get data from a specific column."""
        if column_name not in self.df.columns:
            return f"Column '{column_name}' not found. Available columns: {', '.join(self.df.columns)}"
        
        values = self.df[column_name].tolist()
        return str(values)
    
    def query_data(self, query: str) -> str:
        """Run a pandas query on the dataframe."""
        try:
            result = self.df.query(query)
            if len(result) > 10:
                return f"Query returned {len(result)} rows. First 10 rows:\n{result.head(10).to_string()}"
            return result.to_string()
        except Exception as e:
            return f"Error executing query: {str(e)}"
    
    def filter_data(self, column: str, value: str, operator: str = "==") -> str:
        """Filter dataframe based on column value."""
        if column not in self.df.columns:
            return f"Column '{column}' not found. Available columns: {', '.join(self.df.columns)}"
        
        try:
            # Convert value to appropriate type if numeric
            if pd.api.types.is_numeric_dtype(self.df[column]):
                try:
                    value = float(value)
                except ValueError:
                    pass
            
            # Handle different operators
            if operator == "==":
                filtered_df = self.df[self.df[column] == value]
            elif operator == "!=":
                filtered_df = self.df[self.df[column] != value]
            elif operator == ">":
                filtered_df = self.df[self.df[column] > value]
            elif operator == ">=":
                filtered_df = self.df[self.df[column] >= value]
            elif operator == "<":
                filtered_df = self.df[self.df[column] < value]
            elif operator == "<=":
                filtered_df = self.df[self.df[column] <= value]
            elif operator == "contains":
                filtered_df = self.df[self.df[column].astype(str).str.contains(str(value))]
            else:
                return f"Unsupported operator: {operator}"
            
            if len(filtered_df) > 10:
                return f"Filter returned {len(filtered_df)} rows. First 10 rows:\n{filtered_df.head(10).to_string()}"
            return filtered_df.to_string()
        except Exception as e:
            return f"Error filtering data: {str(e)}"
    
    def group_by_data(self, group_cols: str, agg_dict: str) -> str:
        """Group data by columns and aggregate."""
        try:
            group_cols = [col.strip() for col in group_cols.split(",")]
            for col in group_cols:
                if col not in self.df.columns:
                    return f"Column '{col}' not found. Available columns: {', '.join(self.df.columns)}"
            
            # Parse the aggregation dictionary
            import ast
            agg_dict = ast.literal_eval(agg_dict)
            
            result = self.df.groupby(group_cols).agg(agg_dict).reset_index()
            if len(result) > 10:
                return f"Groupby returned {len(result)} rows. First 10 rows:\n{result.head(10).to_string()}"
            return result.to_string()
        except Exception as e:
            return f"Error in group by operation: {str(e)}"
    
    def sort_data(self, columns: str, ascending: bool = True) -> str:
        """Sort dataframe by specified columns."""
        try:
            sort_cols = [col.strip() for col in columns.split(",")]
            for col in sort_cols:
                if col not in self.df.columns:
                    return f"Column '{col}' not found. Available columns: {', '.join(self.df.columns)}"
            
            result = self.df.sort_values(by=sort_cols, ascending=ascending)
            if len(result) > 10:
                return f"Sorted data ({len(result)} rows). First 10 rows:\n{result.head(10).to_string()}"
            return result.to_string()
        except Exception as e:
            return f"Error sorting data: {str(e)}"
    
    def get_statistics(self, columns: str = "all") -> str:
        """Get descriptive statistics for numeric columns."""
        try:
            if columns.lower() == "all":
                numeric_df = self.df.select_dtypes(include=np.number)
                if numeric_df.empty:
                    return "No numeric columns found in the dataset."
                return numeric_df.describe().to_string()
            
            columns = [col.strip() for col in columns.split(",")]
            stats_df = pd.DataFrame()
            
            for col in columns:
                if col not in self.df.columns:
                    return f"Column '{col}' not found. Available columns: {', '.join(self.df.columns)}"
                
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    return f"Column '{col}' is not numeric. Can only get statistics for numeric columns."
                
                stats_df[col] = self.df[col]
            
            return stats_df.describe().to_string()
        except Exception as e:
            return f"Error getting statistics: {str(e)}"
    
    
    
    def chat(self, message: str) -> str:
        """Chat with the agent about the CSV data."""
        response = self.agent.chat(message)
        return response.response
    
agent = CSVAgent(csv_path=os.path.join(UPLOAD_DIR, "daily-for-last-7-days.csv"))

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    global agent
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    df = pd.read_csv(file_path)
    agent = CSVAgent(csv_path=file_path)
    return {"filename": file.filename, "columns": df.columns.tolist()}

class TextRequest(BaseModel):
    query: str

@app.post("/echo")
async def echo_string(request: TextRequest):
    global agent
    if(agent):
        response = agent.chat(request.query)
        return {"message": f"{response}"}
    
    return {"message":"Agent not initialized reupload csv"}
