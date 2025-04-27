import base64
import sys
from pathlib import Path
import traceback
from typing import List, Optional, Tuple, Dict
from datetime import datetime

import click
import inquirer
import yaml
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import re
from src.libs.resume_and_cover_builder import ResumeFacade, ResumeGenerator, StyleManager
from src.resume_schemas.job_application_profile import JobApplicationProfile
from src.resume_schemas.resume import Resume
from src.logging import logger
from src.utils.chrome_utils import init_browser
from src.utils.constants import (
    PLAIN_TEXT_RESUME_YAML,
    SECRETS_YAML,
    WORK_PREFERENCES_YAML,
)
# from ai_hawk.bot_facade import AIHawkBotFacade
# from ai_hawk.job_manager import AIHawkJobManager
# from ai_hawk.llm.llm_manager import GPTAnswerer


class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass


class ConfigValidator:
    """Validates configuration and secrets YAML files."""

    EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    REQUIRED_CONFIG_KEYS = {
        "remote": bool,
        "experience_level": dict,
        "job_types": dict,
        "date": dict,
        "positions": list,
        "locations": list,
        "location_blacklist": list,
        "distance": int,
        "company_blacklist": list,
        "title_blacklist": list,
    }
    EXPERIENCE_LEVELS = [
        "internship",
        "entry",
        "associate",
        "mid_senior_level",
        "director",
        "executive",
    ]
    JOB_TYPES = [
        "full_time",
        "contract",
        "part_time",
        "temporary",
        "internship",
        "other",
        "volunteer",
    ]
    DATE_FILTERS = ["all_time", "month", "week", "24_hours"]
    APPROVED_DISTANCES = {0, 5, 10, 25, 50, 100}

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate the format of an email address."""
        return bool(ConfigValidator.EMAIL_REGEX.match(email))

    @staticmethod
    def load_yaml(yaml_path: Path) -> dict:
        """Load and parse a YAML file."""
        try:
            with open(yaml_path, "r") as stream:
                return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Error reading YAML file {yaml_path}: {exc}")
        except FileNotFoundError:
            raise ConfigError(f"YAML file not found: {yaml_path}")

    @classmethod
    def validate_config(cls, config_yaml_path: Path) -> dict:
        """Validate the main configuration YAML file."""
        parameters = cls.load_yaml(config_yaml_path)
        # Check for required keys and their types
        for key, expected_type in cls.REQUIRED_CONFIG_KEYS.items():
            if key not in parameters:
                if key in ["company_blacklist", "title_blacklist", "location_blacklist"]:
                    parameters[key] = []
                else:
                    raise ConfigError(f"Missing required key '{key}' in {config_yaml_path}")
            elif not isinstance(parameters[key], expected_type):
                if key in ["company_blacklist", "title_blacklist", "location_blacklist"] and parameters[key] is None:
                    parameters[key] = []
                else:
                    raise ConfigError(
                        f"Invalid type for key '{key}' in {config_yaml_path}. Expected {expected_type.__name__}."
                    )
        cls._validate_experience_levels(parameters["experience_level"], config_yaml_path)
        cls._validate_job_types(parameters["job_types"], config_yaml_path)
        cls._validate_date_filters(parameters["date"], config_yaml_path)
        cls._validate_list_of_strings(parameters, ["positions", "locations"], config_yaml_path)
        cls._validate_distance(parameters["distance"], config_yaml_path)
        cls._validate_blacklists(parameters, config_yaml_path)
        return parameters

    @classmethod
    def _validate_experience_levels(cls, experience_levels: dict, config_path: Path):
        """Ensure experience levels are booleans."""
        for level in cls.EXPERIENCE_LEVELS:
            if not isinstance(experience_levels.get(level), bool):
                raise ConfigError(
                    f"Experience level '{level}' must be a boolean in {config_path}"
                )

    @classmethod
    def _validate_job_types(cls, job_types: dict, config_path: Path):
        """Ensure job types are booleans."""
        for job_type in cls.JOB_TYPES:
            if not isinstance(job_types.get(job_type), bool):
                raise ConfigError(
                    f"Job type '{job_type}' must be a boolean in {config_path}"
                )

    @classmethod
    def _validate_date_filters(cls, date_filters: dict, config_path: Path):
        """Ensure date filters are booleans."""
        for date_filter in cls.DATE_FILTERS:
            if not isinstance(date_filters.get(date_filter), bool):
                raise ConfigError(
                    f"Date filter '{date_filter}' must be a boolean in {config_path}"
                )

    @classmethod
    def _validate_list_of_strings(cls, parameters: dict, keys: list, config_path: Path):
        """Ensure specified keys are lists of strings."""
        for key in keys:
            if not all(isinstance(item, str) for item in parameters[key]):
                raise ConfigError(
                    f"'{key}' must be a list of strings in {config_path}"
                )

    @classmethod
    def _validate_distance(cls, distance: int, config_path: Path):
        """Validate the distance value."""
        if distance not in cls.APPROVED_DISTANCES:
            raise ConfigError(
                f"Invalid distance value '{distance}' in {config_path}. Must be one of: {cls.APPROVED_DISTANCES}"
            )

    @classmethod
    def _validate_blacklists(cls, parameters: dict, config_path: Path):
        """Ensure blacklists are lists."""
        for blacklist in ["company_blacklist", "title_blacklist", "location_blacklist"]:
            if not isinstance(parameters.get(blacklist), list):
                raise ConfigError(
                    f"'{blacklist}' must be a list in {config_path}"
                )
            if parameters[blacklist] is None:
                parameters[blacklist] = []

    @staticmethod
    def validate_secrets(secrets_yaml_path: Path) -> str:
        """Validate the secrets YAML file and retrieve the LLM API key."""
        secrets = ConfigValidator.load_yaml(secrets_yaml_path)
        mandatory_secrets = ["llm_api_key"]

        for secret in mandatory_secrets:
            if secret not in secrets:
                raise ConfigError(f"Missing secret '{secret}' in {secrets_yaml_path}")

            if not secrets[secret]:
                raise ConfigError(f"Secret '{secret}' cannot be empty in {secrets_yaml_path}")

        return secrets["llm_api_key"]


class FileManager:
    """Handles file system operations and validations."""

    REQUIRED_FILES = [SECRETS_YAML, WORK_PREFERENCES_YAML, PLAIN_TEXT_RESUME_YAML]

    @staticmethod
    def validate_data_folder(app_data_folder: Path) -> Tuple[Path, Path, Path, Path]:
        """Validate the existence of the data folder and required files."""
        if not app_data_folder.is_dir():
            raise FileNotFoundError(f"Data folder not found: {app_data_folder}")

        missing_files = [file for file in FileManager.REQUIRED_FILES if not (app_data_folder / file).exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing files in data folder: {', '.join(missing_files)}")

        output_folder = app_data_folder / "output"
        output_folder.mkdir(exist_ok=True)

        return (
            app_data_folder / SECRETS_YAML,
            app_data_folder / WORK_PREFERENCES_YAML,
            app_data_folder / PLAIN_TEXT_RESUME_YAML,
            output_folder,
        )

    @staticmethod
    def get_uploads(plain_text_resume_file: Path) -> Dict[str, Path]:
        """Convert resume file paths to a dictionary."""
        if not plain_text_resume_file.exists():
            raise FileNotFoundError(f"Plain text resume file not found: {plain_text_resume_file}")

        uploads = {"plainTextResume": plain_text_resume_file}

        return uploads


# Action and style mappings
ACTION_MAPPING = {
    "resume": "Generate Resume",
    "job": "Generate Resume Tailored for Job Description",
    "cover": "Generate Tailored Cover Letter for Job Description"
}

STYLE_MAPPING = {
    "clean-blue": "Clean Blue",
    "modern-blue": "Modern Blue",
    "modern-grey": "Modern Grey",
    "default": "Default",
    "cloyola-grey": "Cloyola Grey"
}

def prompt_user_action() -> str:
    """
    Use inquirer to ask the user which action they want to perform.

    :return: Selected action.
    """
    try:
        questions = [
            inquirer.List(
                'action',
                message="Select the action you want to perform:",
                choices=list(ACTION_MAPPING.values()),
            ),
        ]
        answer = inquirer.prompt(questions)
        if answer is None:
            print("No answer provided. The user may have interrupted.")
            return ""
        return answer.get('action', "")
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def prompt_style_selection(style_manager):
    """
    Prompt the user to select a resume style.
    
    :param style_manager: The StyleManager instance
    :return: The selected style name or None if no selection
    """
    available_styles = style_manager.get_styles()
    if not available_styles:
        logger.warning("No styles available. Proceeding without style selection.")
        return None
    
    # Present style choices to the user
    choices = style_manager.format_choices(available_styles)
    questions = [
        inquirer.List(
            "style",
            message="Select a style for the resume:",
            choices=choices,
        )
    ]
    style_answer = inquirer.prompt(questions)
    if style_answer and "style" in style_answer:
        selected_choice = style_answer["style"]
        for style_name, (file_name, author_link) in available_styles.items():
            if selected_choice.startswith(style_name):
                return style_name
    return None

def get_output_filename(action, style, custom_filename=None):
    """
    Generate appropriate output filename based on parameters.
    
    :param action: The action being performed
    :param style: The style selected
    :param custom_filename: Optional custom filename
    :return: The output filename
    """
    if custom_filename:
        return custom_filename
    
    # Map action to short name for filename
    action_name = next((k for k, v in ACTION_MAPPING.items() if v == action), "output")
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{action_name}_{style}_{timestamp}.pdf"

def create_resume_pdf(parameters: dict, llm_api_key: str, style_name=None, output_filename=None):
    """
    Logic to create a CV.
    """
    try:
        logger.info("Generating a CV based on provided parameters.")

        # Load the plain text resume
        with open(parameters["uploads"]["plainTextResume"], "r", encoding="utf-8") as file:
            plain_text_resume = file.read()

        # Initialize StyleManager
        style_manager = StyleManager()
        
        # Set style if provided, otherwise prompt
        if style_name and style_name in STYLE_MAPPING.keys():
            style_manager.set_selected_style(style_name)
            logger.info(f"Using selected style: {style_name}")
        else:
            selected_style = prompt_style_selection(style_manager)
            if selected_style:
                style_manager.set_selected_style(selected_style)
                style_name = selected_style
            else:
                logger.warning("No style selected. Proceeding with default style.")
                style_name = "default"

        # Initialize the Resume Generator
        resume_generator = ResumeGenerator()
        resume_object = Resume(plain_text_resume)
        driver = init_browser()
        resume_generator.set_resume_object(resume_object)

        # Create the ResumeFacade
        resume_facade = ResumeFacade(
            api_key=llm_api_key,
            style_manager=style_manager,
            resume_generator=resume_generator,
            resume_object=resume_object,
            output_path=Path("data_folder/output"),
        )
        resume_facade.set_driver(driver)
        result_base64 = resume_facade.create_resume_pdf()

        # Decode Base64 to binary data
        try:
            pdf_data = base64.b64decode(result_base64)
        except base64.binascii.Error as e:
            logger.error("Error decoding Base64: %s", e)
            raise

        # Define the output filename
        if not output_filename:
            output_filename = get_output_filename("Generate Resume", style_name)
        
        # Define the output directory
        output_dir = Path(parameters["outputFileDirectory"])

        # Write the PDF file
        output_path = output_dir / output_filename
        try:
            with open(output_path, "wb") as file:
                file.write(pdf_data)
            logger.info(f"Resume saved at: {output_path}")
        except IOError as e:
            logger.error("Error writing file: %s", e)
            raise
    except Exception as e:
        logger.exception(f"An error occurred while creating the CV: {e}")
        raise

def create_resume_pdf_job_tailored(parameters: dict, llm_api_key: str, style_name=None, job_url=None, output_filename=None):
    """
    Logic to create a job-tailored CV.
    """
    try:
        logger.info("Generating a job-tailored CV based on provided parameters.")

        # Load the plain text resume
        with open(parameters["uploads"]["plainTextResume"], "r", encoding="utf-8") as file:
            plain_text_resume = file.read()

        # Initialize StyleManager
        style_manager = StyleManager()
        
        # Set style if provided, otherwise prompt
        if style_name and style_name in STYLE_MAPPING.keys():
            style_manager.set_selected_style(style_name)
            logger.info(f"Using selected style: {style_name}")
        else:
            selected_style = prompt_style_selection(style_manager)
            if selected_style:
                style_manager.set_selected_style(selected_style)
                style_name = selected_style
            else:
                logger.warning("No style selected. Proceeding with default style.")
                style_name = "default"
                
        # Prompt for job URL if not provided
        if not job_url:
            questions = [inquirer.Text('job_url', message="Please enter the URL of the job description:")]
            answers = inquirer.prompt(questions)
            job_url = answers.get('job_url')
            
        # Initialize components
        resume_generator = ResumeGenerator()
        resume_object = Resume(plain_text_resume)
        driver = init_browser()
        resume_generator.set_resume_object(resume_object)
        
        # Create facade and process
        resume_facade = ResumeFacade(            
            api_key=llm_api_key,
            style_manager=style_manager,
            resume_generator=resume_generator,
            resume_object=resume_object,
            output_path=Path("data_folder/output"),
        )
        resume_facade.set_driver(driver)
        resume_facade.link_to_job(job_url)
        result_base64, suggested_name = resume_facade.create_resume_pdf_job_tailored()         

        # Decode Base64 to binary data
        try:
            pdf_data = base64.b64decode(result_base64)
        except base64.binascii.Error as e:
            logger.error("Error decoding Base64: %s", e)
            raise

        # Define the output filename
        if not output_filename:
            output_filename = get_output_filename("Generate Resume Tailored for Job Description", style_name)
        
        # Define the output directory
        output_dir = Path(parameters["outputFileDirectory"]) / suggested_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the PDF file
        output_path = output_dir / output_filename
        try:
            with open(output_path, "wb") as file:
                file.write(pdf_data)
            logger.info(f"Resume saved at: {output_path}")
        except IOError as e:
            logger.error("Error writing file: %s", e)
            raise
    except Exception as e:
        logger.exception(f"An error occurred while creating the CV: {e}")
        raise

def create_cover_letter(parameters: dict, llm_api_key: str, style_name=None, job_url=None, output_filename=None):
    """
    Logic to create a CV.
    """
    try:
        logger.info("Generating a cover letter based on provided parameters.")

        # Load the plain text resume
        with open(parameters["uploads"]["plainTextResume"], "r", encoding="utf-8") as file:
            plain_text_resume = file.read()

        # Initialize StyleManager
        style_manager = StyleManager()
        
        # Set style if provided, otherwise prompt
        if style_name and style_name in STYLE_MAPPING.keys():
            style_manager.set_selected_style(style_name)
            logger.info(f"Using selected style: {style_name}")
        else:
            selected_style = prompt_style_selection(style_manager)
            if selected_style:
                style_manager.set_selected_style(selected_style)
                style_name = selected_style
            else:
                logger.warning("No style selected. Proceeding with default style.")
                style_name = "default"
                
        # Prompt for job URL if not provided
        if not job_url:
            questions = [inquirer.Text('job_url', message="Please enter the URL of the job description:")]
            answers = inquirer.prompt(questions)
            job_url = answers.get('job_url')
            
        # Initialize components
        resume_generator = ResumeGenerator()
        resume_object = Resume(plain_text_resume)
        driver = init_browser()
        resume_generator.set_resume_object(resume_object)
        
        # Create facade and process
        resume_facade = ResumeFacade(            
            api_key=llm_api_key,
            style_manager=style_manager,
            resume_generator=resume_generator,
            resume_object=resume_object,
            output_path=Path("data_folder/output"),
        )
        resume_facade.set_driver(driver)
        resume_facade.link_to_job(job_url)
        result_base64, suggested_name = resume_facade.create_cover_letter()         

        # Decode Base64 to binary data
        try:
            pdf_data = base64.b64decode(result_base64)
        except base64.binascii.Error as e:
            logger.error("Error decoding Base64: %s", e)
            raise

        # Define the output filename
        if not output_filename:
            output_filename = get_output_filename("Generate Tailored Cover Letter for Job Description", style_name)
        
        # Define the output directory
        output_dir = Path(parameters["outputFileDirectory"]) / suggested_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write the PDF file
        output_path = output_dir / output_filename
        try:
            with open(output_path, "wb") as file:
                file.write(pdf_data)
            logger.info(f"Cover letter saved at: {output_path}")
        except IOError as e:
            logger.error("Error writing file: %s", e)
            raise
    except Exception as e:
        logger.exception(f"An error occurred while creating the cover letter: {e}")
        raise
    
def handle_inquiries(selected_actions: str, parameters: dict, llm_api_key: str, style=None, job_url=None, output_filename=None):
    """
    Decide which function to call based on the selected user actions.

    :param selected_actions: Action selected by the user.
    :param parameters: Configuration parameters dictionary.
    :param llm_api_key: API key for the language model.
    :param style: Optional style name to use.
    :param job_url: Optional job URL for job-specific documents.
    :param output_filename: Optional custom filename for the output.
    """
    try:
        if selected_actions:
            if "Generate Resume" == selected_actions:
                logger.info("Crafting a standout professional resume...")
                create_resume_pdf(parameters, llm_api_key, style, output_filename)
                
            if "Generate Resume Tailored for Job Description" == selected_actions:
                logger.info("Customizing your resume to enhance your job application...")
                create_resume_pdf_job_tailored(parameters, llm_api_key, style, job_url, output_filename)
                
            if "Generate Tailored Cover Letter for Job Description" == selected_actions:
                logger.info("Designing a personalized cover letter to enhance your job application...")
                create_cover_letter(parameters, llm_api_key, style, job_url, output_filename)
        else:
            logger.warning("No actions selected. Nothing to execute.")
    except Exception as e:
        logger.exception(f"An error occurred while handling inquiries: {e}")
        raise

@click.command()
@click.option('--action', type=click.Choice(['resume', 'job', 'cover']), 
              help='Action to perform: resume=Generate Resume, job=Generate Resume Tailored for Job Description, cover=Generate Tailored Cover Letter')
@click.option('--style', type=click.Choice(['clean-blue', 'modern-blue', 'modern-grey', 'default', 'cloyola-grey']), 
              help='Resume style to use')
@click.option('--job-url', help='URL of the job description for tailored documents')
@click.option('--filename', help='Custom output filename')
def main(action=None, style=None, job_url=None, filename=None):
    """Main entry point for the AIHawk Job Application Bot."""
    try:
        # Define and validate the data folder
        data_folder = Path("data_folder")
        secrets_file, config_file, plain_text_resume_file, output_folder = FileManager.validate_data_folder(data_folder)

        # Validate configuration and secrets
        config = ConfigValidator.validate_config(config_file)
        llm_api_key = ConfigValidator.validate_secrets(secrets_file)

        # Prepare parameters
        config["uploads"] = FileManager.get_uploads(plain_text_resume_file)
        config["outputFileDirectory"] = output_folder

        # Get action from CLI or prompt user
        selected_action = None
        if action:
            selected_action = ACTION_MAPPING.get(action)
            if not selected_action:
                logger.error(f"Invalid action: {action}")
                return
        else:
            # Interactive prompt for user to select actions
            selected_action = prompt_user_action()

        # Handle selected actions and execute them
        handle_inquiries(selected_action, config, llm_api_key, style, job_url, filename)

    except ConfigError as ce:
        logger.error(f"Configuration error: {ce}")
        logger.error(
            "Refer to the configuration guide for troubleshooting: "
            "https://github.com/feder-cr/Auto_Jobs_Applier_AIHawk?tab=readme-ov-file#configuration"
        )
    except FileNotFoundError as fnf:
        logger.error(f"File not found: {fnf}")
        logger.error("Ensure all required files are present in the data folder.")
    except RuntimeError as re:
        logger.error(f"Runtime error: {re}")
        logger.debug(traceback.format_exc())
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
