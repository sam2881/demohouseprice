
### How to Use the Application

**Introduction:**
This guide provides detailed instructions on how to use the PySpark Schema Processing Application to convert JSON schemas into PySpark schemas and generate PySpark scripts to process JSON data.

**Prerequisites:**
- Python 3.x installed on your system.
- Required Python packages: `pyspark`, `pyyaml`.

**Setup Instructions:**

1. **Clone the Repository:**
   - Open your terminal or command prompt.
   - Run the following command to clone the repository:
     ```sh
     git clone https://github.com/your-repository.git
     cd your-repository
     ```

2. **Install Required Packages:**
   - Run the following command to install the necessary Python packages:
     ```sh
     pip install pyspark pyyaml
     ```

3. **Configure the Application:**
   - Create a `config.yaml` file in the root directory of the cloned repository.
   - Add the following configuration details to the `config.yaml` file:
     ```yaml
     json_schema_path: path/to/your/json_schema.json
     pyspark_schema_path: path/to/output/pyspark_schema.py
     pyspark_script_output_path: path/to/output/pyspark_script.py
     json_data_input_path: path/to/input/json_data.json
     ```
   - Replace the placeholder paths with the actual paths to your JSON schema, the desired output location for the PySpark schema and script files, and the path to your JSON data file.

4. **Run the Application:**
   - In your terminal or command prompt, navigate to the root directory of the cloned repository.
   - Run the following command to execute the application:
     ```sh
     python your_script_name.py
     ```
   - The application will generate the PySpark schema file and PySpark script file in the specified output paths.

**Detailed Description of Functions:**

1. **debug_df(df, step_desc):**
   - Prints the schema and data of the DataFrame for debugging purposes.
   - Args:
     - `df` (DataFrame): The DataFrame to debug.
     - `step_desc` (str): Description of the current step in processing.

2. **read_json_file(file_path):**
   - Reads and returns the JSON schema from a file.
   - Args:
     - `file_path` (str): Path to the JSON file.
   - Returns:
     - `dict`: JSON schema as a dictionary.

3. **camel_to_snake(name):**
   - Converts a camelCase string to a snake_case string.
   - Args:
     - `name` (str): The camelCase string.
   - Returns:
     - `str`: The snake_case string.

4. **generate_pyspark_schema(json_schema):**
   - Generates a PySpark schema from a JSON schema dictionary.
   - Args:
     - `json_schema` (dict): JSON schema dictionary.
   - Returns:
     - `StructType`: PySpark StructType schema.

5. **generate_schema_file(schema, schema_file_path):**
   - Generates a Python file containing the schema definition.
   - Args:
     - `schema` (StructType): PySpark StructType schema.
     - `schema_file_path` (str): Path to the output Python file.

6. **extract_column_paths(schema, parent_path=""):**
   - Extracts column paths and explode paths from the schema.
   - Args:
     - `schema` (StructType): PySpark StructType schema.
     - `parent_path` (str): Path prefix for nested fields.
   - Returns:
     - `tuple`: A tuple containing column paths and explode paths.

7. **update_column_paths(column_paths, explode_paths):**
   - Updates column paths with exploded column names.
   - Args:
     - `column_paths` (List[str]): A list of original column paths.
     - `explode_paths` (List[Tuple[str, str]]): A list of tuples (original_path, exploded_path).
   - Returns:
     - `List[str]`: Updated column paths after explosions.

8. **generate_select_expression_from_list(column_names, explode_paths):**
   - Generates a DataFrame select expression with properly aliased columns from a list of column names.
   - Args:
     - `column_names` (List[str]): List of column paths.
     - `explode_paths` (List[tuple]): List of explode paths (original_path, exploded_path).
   - Returns:
     - `str`: Comma-separated select expressions.

9. **generate_pyspark_script(config, schema):**
   - Generates a PySpark script based on the given configuration and schema.
   - Args:
     - `config` (dict): Configuration dictionary.
     - `schema` (StructType): PySpark StructType schema.

10. **Main Execution:**
    - Reads the configuration from `config.yaml`.
    - Reads the JSON schema from the specified path.
    - Generates the PySpark schema.
    - Generates the PySpark script based on the configuration and schema.

**Error Handling:**
- The application includes error handling to log errors during different stages of execution. If an error occurs, it will be logged, and the exception will be raised to stop execution.

**Conclusion:**
This guide provides a comprehensive overview of how to set up and use the PySpark Schema Processing Application. By following the steps outlined above, you will be able to generate PySpark schemas and scripts to process JSON data efficiently. If you encounter any issues or have any questions, please refer to the logging output for debugging information.
