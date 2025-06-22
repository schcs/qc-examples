from qiskit_ibm_runtime import QiskitRuntimeService
import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables

token = os.environ['API_TOKEN']

QiskitRuntimeService.save_account(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
    token=token,
    overwrite=True
)

service = QiskitRuntimeService()

print(service.instances())