"""
Label Classification Module

This module implements a text classification system using AI models to categorize text
into predefined labels. It supports both large and small datasets and uses a language
model to perform the classification tasks.

Key features:
- Automated text classification using AI models
- Support for custom label sets
- Batch processing capabilities
- Detailed output logging and error handling
- Configurable testing modes

The system loads pre-generated labels and classifies new text inputs according to
these existing categories.
"""

import os
import json
import argparse
import time
from dotenv import load_dotenv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT))

from src import api

load_dotenv()


def chat(prompt: str, client) -> str:
    """
    Send a chat request to the Mistral AI model and get classification response.

    Args:
        prompt (str): The formatted input prompt containing text and instructions
        client: The initialized Mistral model client

    Returns:
        str: The model's response containing the classification result

    Note:
        Uses system prompt to ensure JSON output format
        Configured specifically for the Mistral model
    """
    completion = client.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    # response_origin = completion.choices[0].message.content
    # print(f"Original response: {response_origin}")
    return completion


def load_dataset(data_path, data, use_large):
    """
    Load and parse the dataset from JSONL files.

    Args:
        data_path (str): Base path to the data directory
        data (str): Dataset subdirectory name (e.g., 'arxiv_fine')
        use_large (bool): If True, loads large.jsonl, otherwise small.jsonl

    Returns:
        list: List of dictionaries containing parsed JSON objects

    Note:
        Prints dataset path and size for verification
        Expects JSONL format with one JSON object per line
    """
    data_file = (
        os.path.join(data_path, data, "large.jsonl")
        if use_large
        else os.path.join(data_path, data, "small.jsonl")
    )
    print(f"Use dataset {data_file}")
    with open(data_file, "r") as f:
        data_list = []
        for line in f:
            json_object = json.loads(line)
            data_list.append(json_object)
    print(f"Length of dataset: {len(data_list)}")
    return data_list


def get_label_list(data_list):
    label_list = []
    for data in data_list:
        if data["label"] not in label_list:
            label_list.append(data["label"])
    return label_list


def get_predict_labels(output_path, data):
    """
    Load previously generated and merged labels from output files.

    Args:
        output_path (str): Path to the processed data directory
        data (str): Dataset identifier (e.g., 'arxiv_fine')

    Returns:
        list: Deduplicated list of predicted labels

    Note:
        Expects a specific file naming pattern:
        {data}_small_llm_generated_labels_after_merge.json
    """

    data_file = os.path.join(
        output_path, data + "_small_llm_generated_labels_after_merge.json"
    )

    with open(data_file, "r") as f:
        data_list = json.load(f)
    data_list = list(set(data_list))
    return data_list


def prompt_construct(label_list, sentence):
    """
    Construct a classification prompt for the AI model.

    Creates a structured prompt that includes:
    1. Task instruction
    2. Available labels
    3. Input sentence
    4. Expected JSON response format

    Args:
        label_list (list): Available labels for classification
        sentence (str): Text to be classified

    Returns:
        str: Formatted prompt string with complete instructions
    """
    prompt = f"Given the label list and the sentence, please categorize the sentence into one of the labels.\n"
    prompt += f"Label list: {label_list}.\n"
    prompt += f"Sentence:{sentence}.\n"
    json_example = {"label_name": "label"}
    prompt += f"You should only return the label name, please return in json format like: {json_example}"
    return prompt


def answer_process(response, label_list):
    """
    Process and validate the model's response.

    Handles multiple response formats and extracts valid labels:
    1. Attempts to parse response as Python dictionary
    2. Falls back to string processing if parsing fails
    3. Validates extracted label against provided label list

    Args:
        response (str): Raw response from the AI model
        label_list (list): List of valid labels

    Returns:
        str: Valid label if found, "Unsuccessful" otherwise
    """
    label = "Unsuccessful"
    try:
        response_new = eval(response)
    except:
        response_new = str(response)
    if isinstance(response_new, dict):
        total_label = []
        for key, value in response_new.items():
            total_label.append(value)
        for i in label_list:
            if i in total_label:
                label = i
                return label
    else:
        for i in label_list:
            if i in response_new:
                label = i
                return label
    return label


def known_label_categorize(args, client, data_list, label_list):
    """
    Perform batch classification of texts using the AI model.

    Main classification pipeline that:
    1. Processes each input text
    2. Generates classification prompts
    3. Gets and validates model responses
    4. Organizes results by label
    5. Handles errors and unsuccessful classifications

    Args:
        args: Command line arguments with configuration
        client: Initialized AI model client
        data_list (list): Input texts to classify
        label_list (list): Available classification labels

    Returns:
        dict: Mapping of labels to lists of classified texts

    Note:
        Includes progress tracking and periodic result saving
        Supports detailed output printing for debugging
    """
    answer = dict()
    length = args.test_num if args.print_details else len(data_list)
    answer["Unsuccessful"] = []
    for label in label_list:
        answer[label] = []
    for i in range(length):
        sentence = data_list[i]["input"]
        prompt = prompt_construct(label_list, sentence)
        response = chat(prompt, client)
        if response is None:
            response_adjusted = "Unsuccessful"
        else:
            response_adjusted = answer_process(response, label_list)

        if response_adjusted in label_list:
            answer[response_adjusted].append(sentence)
        else:
            answer["Unsuccessful"].append(sentence)

        if args.print_details:
            print(f"---------------Sample {i + 1}-------------------")
            print(f"Question: {sentence}")
            print(f"prompt:\n{prompt}")
            print(f"Original Answer: {response}")
            print(f"Final Answer: {response_adjusted}")
            print(answer)
        if i % 200 == 0:
            print(f"Total sample number: {i}", end="\t")
            write_answer_to_json(args, answer, args.output_path, args.output_file_name)
    return answer


def write_answer_to_json(args, answer, output_path, output_name):
    """
    Save classification results to a JSON file.

    Args:
        args: Command line arguments containing dataset info
        answer (dict): Classification results mapping labels to texts
        output_path (str): Directory for output file
        output_name (str): Base name for output file

    Note:
        Creates filename using pattern: {data}_{size}_{output_name}
        Uses indented JSON format for readability
        Prints confirmation message after writing
    """
    size = "large" if args.use_large else "small"
    file_name = os.path.join(output_path, "_".join([args.data, size, output_name]))
    with open(file_name, "w") as json_file:
        json.dump(answer, json_file, indent=2)
    print(f"JSON file '{file_name}' written.")


def load_predict_data(data_path, file_name):
    data_file = os.path.join(data_path, file_name)
    with open(data_file, "r") as f:
        data_dict = json.load(f)
    return data_dict


def describe_final_output(answer):
    """
    Print summary statistics of classification results.

    Args:
        answer (dict): Classification results mapping labels to texts

    Note:
        Prints count of texts assigned to each label
        Helps verify distribution of classifications
    """
    for key in answer.keys():
        print(f"{key}: {len(answer[key])}")


def main(args):  # given label classification
    """
    Main execution function for the classification pipeline.

    Orchestrates the complete classification process:
    1. Initializes AI model client
    2. Loads dataset and predicted labels
    3. Performs classification on all texts
    4. Removes empty categories
    5. Saves and summarizes results

    Args:
        args: Command line arguments with all configuration parameters

    Note:
        Tracks and reports total execution time
        Prints classification distribution summary
    """
    print(args.use_large)
    start_time = time.time()
    client = api.main("llama")
    data_list = load_dataset(args.data_path, args.data, args.use_large)
    label_list = get_predict_labels(args.output_path, args.data)
    print(f"Length of label list: {len(label_list)}")
    answer = known_label_categorize(args, client, data_list, label_list)
    answer = {k: v for k, v in answer.items() if len(v) != 0}  # remove empty labels
    write_answer_to_json(args, answer, args.output_path, args.output_file_name)
    print(f"Classification result:")
    describe_final_output(answer)
    print(f"Total time usage: {time.time() - start_time} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Text Classification Tool

        Classifies text inputs into predefined categories using the Mistral AI model.
        Supports both large and small datasets, with configurable output detail levels
        and test modes for development purposes.
        """
    )
    parser.add_argument("--data_path", type=str, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--data", type=str, default="arxiv_fine")
    parser.add_argument(
        "--output_path", type=str, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--output_file_name", type=str, default="find_labels.json")
    parser.add_argument(
        "--use_large",
        action="store_true",
        help="Use large model if set, otherwise use small model",
    )  # True - Large; False - Small
    parser.add_argument("--print_details", type=bool, default=False)  # print details
    parser.add_argument("--test_num", type=int, default=5)  # how many test numbers
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="set the key to your OpenAI Key",
    )
    args = parser.parse_args()
    main(args)
