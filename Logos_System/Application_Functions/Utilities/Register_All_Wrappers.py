# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

from __future__ import annotations
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.External_Enhancements_Registry import register, Wrapper_Info

from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.NLP_Wrapper_Nltk import NLP_Wrapper_Nltk
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.NLP_Wrapper_Spacy import NLP_Wrapper_Spacy
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.NLP_Wrapper_Transformers import NLP_Wrapper_Transformers
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.NLP_Wrapper_Sentence_Transformers import NLP_Wrapper_Sentence_Transformers

from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.ML_Wrapper_Torch import ML_Wrapper_Torch
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.ML_Wrapper_Sklearn import ML_Wrapper_Sklearn
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.ML_Wrapper_Statsmodels import ML_Wrapper_Statsmodels
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.ML_Wrapper_Prophet import ML_Wrapper_Prophet
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.ML_Wrapper_Arch import ML_Wrapper_Arch

from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Graph_Wrapper_Networkx import Graph_Wrapper_Networkx
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Graph_Wrapper_Pyvis import Graph_Wrapper_Pyvis
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Graph_Wrapper_Rdflib import Graph_Wrapper_Rdflib

from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Visualization_Wrapper_Matplotlib import Visualization_Wrapper_Matplotlib
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Visualization_Wrapper_Seaborn import Visualization_Wrapper_Seaborn
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Visualization_Wrapper_Plotly import Visualization_Wrapper_Plotly

from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Cognition_Wrapper_Pgmpy import Cognition_Wrapper_Pgmpy
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Cognition_Wrapper_Micropsi import Cognition_Wrapper_Micropsi
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Cognition_Wrapper_Memgpt import Cognition_Wrapper_Memgpt
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Cognition_Wrapper_Openai import Cognition_Wrapper_Openai
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Cognition_Wrapper_Pydantic import Cognition_Wrapper_Pydantic
from Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Cognition_Wrapper_Sympy import Cognition_Wrapper_Sympy

register(Wrapper_Info(wrapper_id="NLP_nltk", wraps="nltk", role="NLP", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.NLP_Wrapper_Nltk", factory=lambda: NLP_Wrapper_Nltk()))
register(Wrapper_Info(wrapper_id="NLP_spacy", wraps="spacy", role="NLP", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.NLP_Wrapper_Spacy", factory=lambda: NLP_Wrapper_Spacy()))
register(Wrapper_Info(wrapper_id="NLP_transformers", wraps="transformers", role="NLP", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.NLP_Wrapper_Transformers", factory=lambda: NLP_Wrapper_Transformers()))
register(Wrapper_Info(wrapper_id="NLP_sentence_transformers", wraps="sentence-transformers", role="NLP", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.NLP_Wrapper_Sentence_Transformers", factory=lambda: NLP_Wrapper_Sentence_Transformers()))

register(Wrapper_Info(wrapper_id="ML_torch", wraps="torch", role="ML", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.ML_Wrapper_Torch", factory=lambda: ML_Wrapper_Torch()))
register(Wrapper_Info(wrapper_id="ML_sklearn", wraps="scikit-learn", role="ML", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.ML_Wrapper_Sklearn", factory=lambda: ML_Wrapper_Sklearn()))
register(Wrapper_Info(wrapper_id="ML_statsmodels", wraps="statsmodels", role="ML", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.ML_Wrapper_Statsmodels", factory=lambda: ML_Wrapper_Statsmodels()))
register(Wrapper_Info(wrapper_id="ML_prophet", wraps="prophet", role="ML", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.ML_Wrapper_Prophet", factory=lambda: ML_Wrapper_Prophet()))
register(Wrapper_Info(wrapper_id="ML_arch", wraps="arch", role="ML", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.ML_Wrapper_Arch", factory=lambda: ML_Wrapper_Arch()))

register(Wrapper_Info(wrapper_id="Graph_networkx", wraps="networkx", role="Graph", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Graph_Wrapper_Networkx", factory=lambda: Graph_Wrapper_Networkx()))
register(Wrapper_Info(wrapper_id="Graph_pyvis", wraps="pyvis", role="Graph", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Graph_Wrapper_Pyvis", factory=lambda: Graph_Wrapper_Pyvis()))
register(Wrapper_Info(wrapper_id="Graph_rdflib", wraps="rdflib", role="Graph", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Graph_Wrapper_Rdflib", factory=lambda: Graph_Wrapper_Rdflib()))

register(Wrapper_Info(wrapper_id="Visualization_matplotlib", wraps="matplotlib", role="Visualization", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Visualization_Wrapper_Matplotlib", factory=lambda: Visualization_Wrapper_Matplotlib()))
register(Wrapper_Info(wrapper_id="Visualization_seaborn", wraps="seaborn", role="Visualization", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Visualization_Wrapper_Seaborn", factory=lambda: Visualization_Wrapper_Seaborn()))
register(Wrapper_Info(wrapper_id="Visualization_plotly", wraps="plotly", role="Visualization", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Visualization_Wrapper_Plotly", factory=lambda: Visualization_Wrapper_Plotly()))

register(Wrapper_Info(wrapper_id="Cognition_pgmpy", wraps="pgmpy", role="Cognition", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Cognition_Wrapper_Pgmpy", factory=lambda: Cognition_Wrapper_Pgmpy()))
register(Wrapper_Info(wrapper_id="Cognition_micropsi", wraps="micropsi", role="Cognition", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Cognition_Wrapper_Micropsi", factory=lambda: Cognition_Wrapper_Micropsi()))
register(Wrapper_Info(wrapper_id="Cognition_memgpt", wraps="memgpt", role="Cognition", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Cognition_Wrapper_Memgpt", factory=lambda: Cognition_Wrapper_Memgpt()))
register(Wrapper_Info(wrapper_id="Cognition_openai", wraps="openai", role="Cognition", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Cognition_Wrapper_Openai", factory=lambda: Cognition_Wrapper_Openai()))
register(Wrapper_Info(wrapper_id="Cognition_pydantic", wraps="pydantic", role="Cognition", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Cognition_Wrapper_Pydantic", factory=lambda: Cognition_Wrapper_Pydantic()))
register(Wrapper_Info(wrapper_id="Cognition_sympy", wraps="sympy", role="Cognition", module_path="Logos_System.System_Stack.Logos_Protocol.External_Enhancements.Cognition_Wrapper_Sympy", factory=lambda: Cognition_Wrapper_Sympy()))
