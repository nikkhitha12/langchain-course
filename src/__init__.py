# Import load_dotenv to load environment variables from the .env file
from dotenv import load_dotenv

# Import PromptTemplate to create a reusable prompt
from langchain_core.prompts import PromptTemplate

# Import ChatOllama to connect LangChain with the Ollama local LLM
from langchain_ollama import ChatOllama


# Load environment variables
load_dotenv()


# Store information about Elon Musk
info = """Elon Musk is a famous businessman, entrepreneur, and technology leader.
He was born on June 28, 1971, in Pretoria, South Africa.
He developed an interest in computers and technology at a young age.
He later moved to North America to continue his education.
He studied physics and economics at the University of Pennsylvania.
Musk began his business career by co-founding Zip2.
He later founded X.com, an online financial company.
X.com eventually became part of PayPal.
PayPal became one of the successful online payment companies.
In 2002, Musk founded SpaceX.
SpaceX focuses on rockets, spacecraft, and space exploration.
Musk is also associated with Tesla, an electric vehicle company.
He became Tesla's CEO in 2008.
Tesla helped popularize electric cars around the world.
Musk also founded Neuralink.
Neuralink works on brain-computer interface technology.
He founded The Boring Company to work on transportation infrastructure.
In 2022, Musk acquired Twitter and later renamed it X.
Musk also founded the artificial intelligence company xAI.
He has a strong interest in artificial intelligence and robotics.
He is known for setting ambitious goals for his companies.
SpaceX has developed reusable rocket technology.
Musk has played an important role in the development of commercial space technology.
He is one of the most widely recognized technology entrepreneurs in the world.
His companies operate in areas such as electric vehicles, space, AI, and technology.
His business decisions and public statements have also attracted criticism and controversy.
Despite this, he remains an influential figure in the technology industry.
He continues to work on projects related to space exploration, AI, and transportation.
Elon Musk's career demonstrates the significant impact that technology and entrepreneurship can have on modern industries."""


# Create the prompt template
summary_template = """
Given the following information about a person:

{info}

Create:
1. A short summary
2. Three important achievements
"""


# Create a PromptTemplate using the info variable
summary_prompt_template = PromptTemplate(
    input_variables=["info"],
    template=summary_template
)


# Initialize the Ollama language model
llm = ChatOllama(
    model="gemma3:270m",
    temperature=0
)


# Connect the prompt template with the LLM
chain = summary_prompt_template | llm


# Send the Elon Musk information to the chain
res = chain.invoke({"info": info})


# Print the model's response
print(res.content)