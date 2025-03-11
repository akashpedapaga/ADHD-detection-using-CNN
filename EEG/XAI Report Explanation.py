'''
from bs4 import BeautifulSoup
import json
import matplotlib.pyplot as plt
import re

# Load the HTML file
html_file_path = r'F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\lime_explanation.html'
with open(html_file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Find the JavaScript variable 'var lime' that contains the explanation data
script_tag = soup.find('script', string=lambda text: text and 'var lime =' in text)

if script_tag:
    script_content = script_tag.string
    
    # Debugging step: Print the script content
    print("Script Content: ", script_content[:500])  # Print first 500 characters for inspection

    # Extract the JSON data from the JavaScript variable
    try:
        # Use regex to extract JSON data between 'var lime =' and the first semicolon
        json_data_match = re.search(r'var lime = (\{.*?\});', script_content, re.DOTALL)
        if json_data_match:
            json_data = json_data_match.group(1).strip()
            
            # Debugging step: Print the extracted JSON data
            print("Extracted JSON Data: ", json_data[:500])  # Print first 500 characters for inspection
            
            # Load JSON data
            lime_data = json.loads(json_data)
            
            # Assuming lime_data structure and accessing explanation
            explanation = lime_data['explanation']
            
            # Extract feature names and weights for visualization
            features = [(item[0], item[1]) for item in explanation]
            feature_names = [f[0] for f in features]
            weights = [f[1] for f in features]
            
            # Create a bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.barh(feature_names, weights, color=['green' if w > 0 else 'red' for w in weights])
            plt.xlabel('Feature Contribution')
            plt.ylabel('Features')
            plt.title('LIME Feature Contributions')
            
            # Add values on top of the bars
            for bar in bars:
                plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', 
                         va='center', ha='left' if bar.get_width() > 0 else 'right')
            
            plt.show()
        else:
            print("Failed to extract JSON data from the script content.")
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
    except KeyError as e:
        print(f"KeyError: {e}. Ensure the JSON structure matches the expected format.")
else:
    print("Script tag containing 'var lime =' not found in the HTML file.")
'''




from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the HTML content
file_path = 'explanation.html'
with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Function to extract text from divs, spans, and other common elements
def extract_text(elements):
    data = []
    for element in elements:
        text = element.get_text().strip()
        if text:  # Only add non-empty strings
            data.append(text)
    return data

# Extract text from div and span elements as an example
divs = soup.find_all('div')
spans = soup.find_all('span')

div_texts = extract_text(divs)
span_texts = extract_text(spans)

# Combine the data
combined_data = div_texts + span_texts

# Print extracted data for inspection
print("Extracted Data:")
for item in combined_data:
    print(item)

# Create a DataFrame from the extracted data
# Assuming we have two types of data, div_texts and span_texts
# Handling different lengths of data by padding with None
max_length = max(len(div_texts), len(span_texts))
div_texts += [None] * (max_length - len(div_texts))
span_texts += [None] * (max_length - len(span_texts))

df = pd.DataFrame({
    'Div Texts': pd.Series(div_texts),
    'Span Texts': pd.Series(span_texts)
})

# Display the DataFrame using pandas
print("\nDataFrame:")
print(df)

# Example visualizations
# Customize the column names and visualization based on your actual data

# Example Bar Plot for Div Texts
plt.figure(figsize=(10, 6))
sns.barplot(x=df.index, y='Div Texts', data=df)  # Replace 'Div Texts' with the actual column name
plt.title('Bar Plot of Div Texts')
plt.xlabel('Index')
plt.ylabel('Div Texts')
plt.xticks(rotation=90)
plt.show()

# Example Line Plot for Span Texts
plt.figure(figsize=(10, 6))
sns.lineplot(x=df.index, y='Span Texts', data=df)  # Replace 'Span Texts' with the actual column name
plt.title('Line Plot of Span Texts')
plt.xlabel('Index')
plt.ylabel('Span Texts')
plt.xticks(rotation=90)
plt.show()

# Example Heatmap if applicable
# This will only work if your data is in a numeric matrix format
if not df.empty:
    plt.figure(figsize=(10, 6))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

















'''
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the HTML content
file_path = 'F:\Local Disk F\MS CS\Academics\Thesis\Code\Adult-Adhd\EEG\explanation.html'
with open(file_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Extract tables from the HTML content
tables = soup.find_all('table')
data = []

if tables:
    table = tables[0]
    headers = [th.text for th in table.find_all('th')]
    rows = table.find_all('tr')[1:]  # skip header row

    for row in rows:
        cols = row.find_all('td')
        data.append([col.text for col in cols])

    # Create a DataFrame
    if headers:
        df = pd.DataFrame(data, columns=headers)
    else:
        df = pd.DataFrame(data)

    # Display the DataFrame
    import ace_tools as tools; tools.display_dataframe_to_user(name="Extracted Data", dataframe=df)

    # Example visualizations

    # Bar plot of a specific column
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Category', y='Value', data=df)  # Replace 'Category' and 'Value' with actual column names
    plt.title('Bar Plot of Category vs Value')
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.show()

    # Line plot if there's a time series
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Date', y='Value', data=df)  # Replace 'Date' and 'Value' with actual column names
    plt.title('Line Plot of Date vs Value')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()

    # Heatmap for correlation if applicable
    plt.figure(figsize=(10, 6))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

else:
    print("No tables found in the HTML file.")
'''