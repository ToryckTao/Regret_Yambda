#!/usr/bin/env python
"""Test script to verify SASRec_Yambda imports work correctly."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

# Test utils
print("1. Testing utils...")
import utils
print("   utils OK")

# Test env
print("2. Testing env...")
from env import YambdaEnvironment_GPU
print("   YambdaEnvironment_GPU OK")
from env.BaseRLEnvironment import BaseRLEnvironment
print("   BaseRLEnvironment OK")
from env.reward import *
print("   reward OK")

# Test reader
print("3. Testing reader...")
from reader.YambdaDataReader import YambdaDataReader
print("   YambdaDataReader OK")
from reader.BaseReader import BaseReader
print("   BaseReader OK")

# Test model
print("4. Testing model...")
from model import SASRec, DDPG, GeneralCritic, OneStageFacade
print("   SASRec OK")
print("   DDPG OK")
print("   GeneralCritic OK")
print("   OneStageFacade OK")

from model.YambdaUserResponse import YambdaUserResponse
print("   YambdaUserResponse OK")

from model.components import DNN
print("   DNN OK")

from model.score_func import dot_scorer
print("   dot_scorer OK")

print("\n" + "="*50)
print("All imports successful!")
print("="*50)
