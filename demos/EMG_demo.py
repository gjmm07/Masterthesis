from DelsysCentro import DelsysCentro

with DelsysCentro() as centro:
    for _ in range(200):
        print(centro.get_data())



