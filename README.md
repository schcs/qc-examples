# Qiskit Examples

This repository contains examples of how to use qiskit. Each Jupyter Notebook contains one example.

## Learning Material

- IBM Learning Course: https://learning.quantum.ibm.com/course/quantum-computing-in-practice/introduction

## Setup instructions

1. Create your virtual env

```sh
python3 -m venv venv
```

2. Create your IBM account (https://quantum.ibm.com/) and put your token on .env file

```sh
cp .env.example .env
```

3. Save your account info `python3 auth.py`. Your credentials will be stored at `~/.qiskit/qiskit-ibm.json` for future use.

4. For each notebook select the correct kernel (venv python).

## Troubleshooting

- If you have any trouble trying to conect to IBM cloud please check the link: https://docs.quantum.ibm.com/errors.
- To get your instance to connect to ibm cloud go to

```
Profile Icon > Manage Account > Instances > ... > Copy 'qiskit-ibm-runtime' code
```

Example:

```py
from qiskit_ibm_runtime import QiskitRuntimeService

service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token='<IBM Quantum API key>'
)

# Or save your credentials on disk.
# QiskitRuntimeService.save_account(channel='ibm_quantum', instance='ibm-q/open/main', token='<IBM Quantum API key>')
```

- Support (for the Open Plan use the Slack channel): https://docs.quantum.ibm.com/support
