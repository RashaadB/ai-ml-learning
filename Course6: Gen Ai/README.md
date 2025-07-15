## Essentials and applications of generative AI

`intro_gen_ai.ipynb`
- Apply autoencoders to compress and decode image data for improving website loading times in e-commerce
- Utilize GANs to create high-quality synthetic images for augmenting datasets in healthcare applications
- Implement transformers to improve the accuracy of language translation services in multilingual communication platforms
- Comprehend the principles of RAG and explore methods to enhance responses in customer support chatbots through the integration of retrieved and generated information

`vae_tensor_image_generation`
- Set Hyperparameters
- Define Model Architecture
- Define the Sampling Function
- Connect the Encoder and Decoder
- Define the Loss Function and Compile the Model
- Train the Model
- Generate a Manifold of Digits

`gen_fake_images_gans.ipynb`
- Build the Generator and Discriminator
- Compile the Models
- Train the Models
- Execute the Training
- Generate New Images and Evaluate the Model's Performance

## LangChain and LLMs Folder
`intro_langchain.ipynb`
- Language Models
- LLMs
- architecture of LLMs
- LangChain
- LangChain architecture
- Systems requirements for LangChain

`text_generation.ipynb`
- Define Stopwords and Punctuation
- Load Sentences and Generate N-grams
- Remove Stopwords from N-grams
- Calculate Frequency Distributions
- Create a Dictionary of Trigram Frequencies
- Define the Text Generation Function
- Execute the Text Generation Function

`text_generation_pipeline_langchain.ipynb`
- Authenticate the Hugging Face Account and Set the API Key
- Use the Hugging Face Hub to Load the Flan T5 XXL model
- Create a Langchain HuggingFacePipeline for Text Generation
- Build a Chain Using Langchain
- Test and Run the Chain on Few a Questions

`langchain_llm_application.ipynb`
- Model I/O
    - prompts
    - language models
    - parsers
- Document Loaders
- Text splitters
    - CharacterTextSplitter
    - RecursiveTextSplitter
- Embeddings in GenAi
- Text embedding in langChain
- VectorStore
- LangChain retriever
- Langchain chains
- sequenntial chain 
- stuff chain
- refine chain
- map reduce chain
- langChain memory
- LangChain agents

`langchain_prompt_outputparser.ipynb`
- Set up the Environment
- Call Direct API to OpenAI
- Call the Direct API at OpenAI
- Use the Chat Model
- Format a New Message
- Generate a Response in a New Style
- Output Parsers
- Use the Output Parser

`langchain_loader_splitter_embeddings_vectorstore.ipynb`
<!-- In this activity, you will implement the functionalities of LangChain’s loaders, splitters, embeddings, and VectorStores.
The two files in the tutorial serve as practical examples of real-world data that one might encounter in natural language processing tasks. They are:

•	The **state_of_union.txt** file, which contains transcripts of the United States’ State of the Union Addresses, represents a large text document that can be loaded and processed.

•	The **michael_resume.pdf** file, an open source resume, represents a common type of document that one might analyze for tasks such as resume screening or information extraction. -->

`langchain_sequential_chain.ipynb`
1. Import the Necessary Modules
2. Define a Function to Print Responses from Our Chains
3. Initialize the Chat Model
4. Define the First Chain
5. Define the Second Chain
6. Define the Sequential Chain
7. Run the Sequential Chain

`langchain_memory.ipynb`
 Import the Necessary Modules
2. Initialize the Chat Model
3. Define ConversationBufferMemory
4. Define ConversationBufferWindowMemory with a Window Size of 1
5. Define ConversationTokenBufferMemory with a Maximum Token Limit of 30
6. Define ConversationSummaryBufferMemory with a Maximum Token Limit of 100

`langchain_agents.ipynb`
1. Import the Necessary Modules
2. Initialize the Chat Model
3. Load the Tools
4. Initialize Agent
5. Use Agent to Ask Questions
6. Use Agent to Ask Another Question

`run_llm_falcon.ipynb`
1. Set up the Environment
2. Download Falcon 7B Model and Tokenizer from Hugging Face
3. Set up Model and Generation Configuration
3. Build the Conversation Chain
4. Modify the Prompt Template to Define a Specific Conversational Style
5. Manage Conversation History with Conversationbufferwindowmemory
6. Interact with the LLM

## Advanced Prompting Folder
`advanced_prompting.ipynb`
- Prompt Engineering
- Optimizing Basic Prompts
- Advanced Prompt Engineering
- LLM settings for prompting
- Prompt Elements
- Promt Techniques
- CoT prompting
- Self consistency prompting
- ToT prompting
- LangChain prompts

`zero_shot_promting.ipynb`
- Set up the OpenAI API Key
- Define a Function to Get Completion
- Define Your Prompt

`few_shot_promting.ipynb`
- Set up the OpenAI API Key
- Define a Function to Get Completion
- Define Your Prompt

`chain_of_thought.ipynb`
- Set up the OpenAI API Key
- Define a Function to Get Completion

`self_consistence.ipynb`
- Set up the OpenAI API Key
- Define a Function to Get Completion
- Define Your Prompts

`tree_of_thoughts.ipynb`
- Set up the OpenAI API Key
- Define a Function to Get Completion
- Define Your Prompts

`jinja2_template_format.ipynb`
- Define a Template Using the Jinja2 Format
- Create a Prompt Using the Jinja2 Template
- Use the Prompt to Generate a Question

`f_string-templet.ipynb`
- Define a Template Using the f-string Format
- Create a Prompt Using the f-string Template
- Use the Prompt to Generate a Question
- Output the Summary

`custom_templet.ipynb`
- Set up the Environment
- Define the Prompt Template
- Create a Custom Prompt Template Class

`dynamic_message_langchain.ipynb`
- Import the Necessary Components
- Define Message Templates
- Create a Chat Prompt with Placeholders
- Define the Conversation Messages
- Generate the Conversation
- Print the Conversation

## LLM Fine Tuning and Customization Folder
`fine_tuning_intro.ipynb`
- Need of fine tuning
- data preparation
- fine tuning methodologies
- supervised fine tuning
- parameter efficient fine tuning
    - LoRA
    - P tuning (prompt tuning)
    - Prefix tuning
    - adapters
    - AdaLoRA
- reinforcement leanring
- hyperparameter tuning 
- evaluation of fine tuning models
- hands on fine tuning
- fine tuning best practices
- common biases with llm fine tuning
    - spurious correlations and underrepreentation
    - fairness and inductive biases