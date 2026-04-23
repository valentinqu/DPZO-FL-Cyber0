from opacus.accountants import RDPAccountant

TOTAL_SAMPLES = 60000
BATCH_SIZE = 128
ROUNDS = 900
LOCAL_STEPS = 10
NUM_SAMPLE_CLIENTS = 5  

TOTAL_STEPS = ROUNDS * NUM_SAMPLE_CLIENTS * LOCAL_STEPS 
SAMPLE_RATE = BATCH_SIZE / TOTAL_SAMPLES
NOISE_MULTIPLIER = 0.5
DELTA = 1e-5

accountant = RDPAccountant()
accountant.history = [(NOISE_MULTIPLIER, SAMPLE_RATE, TOTAL_STEPS)]
epsilon = accountant.get_epsilon(delta=DELTA)

print(f"Privacy Budget Epsilon (ε) = {epsilon:.4f}")