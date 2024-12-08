from kafka import KafkaProducer
from concurrent.futures import ThreadPoolExecutor
import random
import time

producer = KafkaProducer(bootstrap_servers='localhost:9092')

def simulate_data(producer_id):
    while True:
        duration = random.randint(12, 50)
        age = random.randint(18, 70)
        amount = random.randint(500, 8000)

        message = f"{duration},{age},{amount}"

        producer.send('customer_data', value=message.encode('utf-8'))
        print(f"User-{producer_id} Sent: {message}")

        time.sleep(random.uniform(0.5, 2)) 

num_producers = 3  
with ThreadPoolExecutor(max_workers=num_producers) as executor:
    for producer_id in range(1, num_producers + 1):
        executor.submit(simulate_data, producer_id)
