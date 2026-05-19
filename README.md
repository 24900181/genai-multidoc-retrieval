## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Extracting specific, nuanced information from a collection of dense academic papers is a slow and inefficient manual process. Standard search tools rely on exact keywords and fail to understand the conceptual context of a user's question. This program aims to build an AI agent that can intelligently query multiple documents to synthesize precise answers to complex questions.

### DESIGN STEPS:

#### STEP 1: Load PDF documents and create specialized search and summary tools for each paper.

#### STEP 2: Initialize an AI agent with an OpenAI model, giving it access to all the created tools.

#### STEP 3:  Query the agent with a specific question about one paper to get a detailed answer from its content.

### PROGRAM:
```
NAME : NETHRA.K
REG.NO : 212224230184
```

```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
```

```
import nest_asyncio
nest_asyncio.apply()
```
```
urls = [
    "https://openreview.net/attachment?id=0HRlus3l4l&name=pdf",
    "https://openreview.net/pdf?id=bBF077z8LF",
    "https://openreview.net/attachment?id=kMuQBgPIdg&name=pdf",
    "https://openreview.net/attachment?id=ogMxCjdCCq&name=pdf",
    "https://openreview.net/attachment?id=krGpQzo8Mz&name=pdf",
]

papers = [
    "62_OpenFedLLM_Training_Large_L.pdf",
    "55_Autonomous_Data_Selection_w.pdf",
    "24514_Enhancing_Generative_Aut.pdf",
    "22727_Latent_Fourier_Transform.pdf",
    "22526_Latent_Speech_Text_Trans.pdf",
]
```

```
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
```

```
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
```

```
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
```

```
len(initial_tools)
```

```
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
```

```
response = agent.query(
    "In what ways do latent speech patches help bridge or transfer capabilities between text tokens and speech tokens more effectively than a standard frame-by-frame approach?, "
    "How does the Latent Speech-Text Transformer (LST) dynamically aggregate speech tokens into '''latent speech patches?''' What properties or types of speech sequences (e.g., silences) do these patches capture?"
)
```
### OUTPUT:
<img width="1307" height="342" alt="image" src="https://github.com/user-attachments/assets/7a44c344-56fc-4118-9f46-4e7a75d3145f" />

<img width="1316" height="92" alt="image" src="https://github.com/user-attachments/assets/4041687a-d838-476a-826d-ed5b911ddd9f" />

<img width="1072" height="846" alt="image" src="https://github.com/user-attachments/assets/a046026a-b756-438d-9da1-755b7d0b3f10" />

<img width="1002" height="921" alt="image" src="https://github.com/user-attachments/assets/ca50bd8b-6869-4595-b5a3-8acf8194b219" />

### RESULT:
The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.
