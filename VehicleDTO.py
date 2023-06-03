import json

class VehicleDTO:
    license_plate: str
    brand: str
    apk_expiry_date: str
    veh_registration_nr: str
    
    def __init__(self, license_plate, brand, apk_expiry_date, veh_registration_nr):
        self.license_plate = license_plate
        self.brand = brand
        self.apk_expiry_date = apk_expiry_date
        self.veh_registration_nr = veh_registration_nr

    @staticmethod
    def from_json(data):
        # Access specific values using the keys
        license_plate = data[0]['kenteken']
        brand = data[0]['merk']
        apk_expiry_date = data[0]['vervaldatum_apk_dt']
        veh_registration_nr = data[0]['typegoedkeuringsnummer']   
        return VehicleDTO(license_plate, brand, apk_expiry_date, veh_registration_nr)