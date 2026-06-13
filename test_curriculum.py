from bim4.species import DigitalSpecies

def run_curriculum():
    print("--- SPANNING NEW DIGITAL SPECIES ---\n")
    
    print("[Phase 1: Motor Babbling]")
    baby1 = DigitalSpecies()
    print("Feeding empty space to simulate passing time without input.")
    babbles = []
    for _ in range(20):
        out = baby1.interact(" ")
        babbles.append(repr(out))
    print("Baby outputs:", " ".join(babbles))
    print(f"End of Phase 1 State: {baby1.brainstem.get_state()}\n")
    
    print("[Phase 2: Pavlovian Association]")
    baby2 = DigitalSpecies() # Fresh brain so it doesn't remember Phase 1
    print("We want it to learn that outputting 'B' feels good.")
    print("Training 50 iterations by directly rewarding the 'B' action in the Basal Ganglia...")
    for _ in range(50):
        sdr = baby2.thalamus.encode(ord(' '))
        baby2.basal_ganglia.learn(sdr, ord('B'), 1.0)
        
    print("Now we test: User gives an empty prompt (space).")
    test_out = baby2.interact(" ")
    print(f"User: ' ' -> Baby: {repr(test_out)}")
    print(f"End of Phase 2 State: {baby2.brainstem.get_state()}\n")
    
    print("[Phase 3: Stimulus-Response (A -> B)]")
    baby3 = DigitalSpecies() # Fresh brain
    print("We want it to reliably respond with 'B' ONLY when it sees 'A'.")
    print("Training A->B loop for 50 iterations...")
    for _ in range(50):
        sdr = baby3.thalamus.encode(ord('A'))
        baby3.basal_ganglia.learn(sdr, ord('B'), 1.0)
    
    test_out = baby3.interact("A")
    print(f"User: 'A' -> Baby: {repr(test_out)}\n")
    
    print("[Phase 4: Alphabet Sequence]")
    baby4 = DigitalSpecies() # Fresh brain!
    print("Feeding A B C D E")
    baby4.interact("A")
    baby4.interact("B")
    baby4.interact("C")
    baby4.interact("D")
    baby4.interact("E")
    
    print("Triggering [SLEEP] for consolidation...")
    baby4.interact("[SLEEP]")
    
    print("Waking up. Let's see what happens when we type A.")
    # Because our Cortex predicts the *next* state, we can force_predict to see the chain
    sdr = baby4.thalamus.encode(ord('A'))
    chain = []
    for _ in range(4):
        pred_sdr = baby4.cortex.force_predict(sdr)
        if len(pred_sdr) == 0:
            break
        char = chr(baby4.thalamus.decode(pred_sdr))
        chain.append(char)
        sdr = pred_sdr
        
    print(f"Hippocampal/Cortical consolidated chain starting from A: {''.join(chain)}\n")

if __name__ == "__main__":
    run_curriculum()
