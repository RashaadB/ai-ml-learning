## Incremental Capstone

BikeEase has successfully implemented various AI-powered solutions for demand forecasting, customer review analysis, and image classification. As they continue to grow, they aim to automate certain tasks using Large Language Models (LLMs), particularly in marketing and advertising generation to attract more customers and increase engagement.

To achieve this, BikeEase plans to develop a Generative AI-powered system that can automatically create engaging and persuasive advertisements based on bike specifications, discount offers, and promotional themes. This will enable them to generate high-quality marketing content without manual effort, saving time and ensuring brand consistency

Project Statement

Develop a Generative AI-powered advertisement generation system using LLMs and LangChain to create compelling promotional content for BikeEase’s rental services

Steps to Perform

Task 1: Understand generative AI & LLMs

Explore how LLMs can be used for automated marketing
Learn about LangChain and how it helps integrate LLMs into applications
Task 2: Designing the Ad generation pipeline

Accept user inputs for bike specifications, discount options, and marketing themes
Use LLMs (Hugging Face models) to generate creative, engaging ads
Structure the output to align with BikeEase’s branding and tone
Task 3: Building the LLM-based Ad generator

Use LangChain to manage the prompt engineering process
Integrate a local Hugging Face model to generate text without API dependencies
Experiment with different prompt techniques to enhance response quality
Task 4: Evaluation and optimization

Test the ad variations to ensure quality, persuasiveness, and relevance.
Implement prompt tuning to fine-tune outputs for different use cases.
Compare different LLM models to identify the most effective one for marketing

## End of Course Project
Crafting an AI-Powered HR Assistant: A Use Case for Nestle’s HR Policy Documents
Overview

The project aims to create a conversational chatbot that responds to user inquiries using PDF document information. It requires proficiency in extracting and converting text into numerical vectors, establishing an answer-finding mechanism, and designing a user-friendly chatbot interface with Gradio. Additionally, the initiative emphasizes structuring inquiries for clear communication and deploying the chatbot for practical use, guaranteeing the system's accessibility and efficiency in meeting user needs.

Instructions 

Review the learning materials and the Gradio documentation provided for the project.
Read the sections on situation, task, action, and result carefully to understand the assignment.
Complete and submit the assignment through the Learning Management System (LMS).
Adhere closely to the provided guidelines, ensuring your submission contains all necessary analyses and interpretations.
Situation

As a developer, you have received the critical task of improving the operational efficiency of Nestlé's human resources department, a leading multinational corporation. Your toolkit includes cutting-edge conversational AI technology, Python libraries, the powerful GPT model from OpenAI, and the user-friendly Gradio UI. Your mission is to integrate these advanced tools seamlessly to transform HR processes, creating a more streamlined and efficient workflow within the Nestlé organization.

Task

Your task is to develop a conversational chatbot. This chatbot must answer queries about Nestlé's HR reports efficiently. Use Python libraries, OpenAI's GPT model, and Gradio UI. These tools will help you create a user-friendly interface. This interface will extract and process information from documents. It will provide accurate responses to user queries.

Action

Import essential tools and set up OpenAI's API environment.
Load Nestle's HR policy using PyPDFLoader and split it for easy processing.
Create vector representations for text chunks using Chroma dB and OpenAI's embeddings.
Build a question-answering system using the GPT-3.5 Turbo model to retrieve answers from text chunks.
Create a prompt template to guide the chatbot in understanding and responding to users.
Use Gradio to build a user-friendly chatbot interface, enabling interaction and information retrieval.
Result

Upon completing this project, you will submit an IPYNB file demonstrating your ability to use advanced AI and machine learning technologies to develop a conversational chatbot. Your submission must include the entire workflow: setting up the programming environment, processing text documents, creating text vector representations, and building a question-answering system. Ensure the interface is user-friendly to facilitate effective interaction and information retrieval.