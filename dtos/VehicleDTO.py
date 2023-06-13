class VehicleDTO:
    """
    Data Transfer Object (DTO) for representing vehicle information.

    Attributes:
        license_plate (str): License plate number.
        brand (str): Brand of the vehicle.
        apk_expiry_date (str): Expiry date of the APK (Algemene Periodieke Keuring) inspection.
        veh_registration_nr (str): Type approval number of the vehicle registration.

    Methods:
        __init__(self, license_plate, brand, apk_expiry_date, veh_registration_nr): Initializes a new VehicleDTO instance.
        from_json(data): Creates a VehicleDTO object from JSON data.
        brand_dto(data): Creates a VehicleDTO object with only the brand information.

    """
    license_plate: str
    brand: str
    apk_expiry_date: str
    veh_registration_nr: str
    """
        Initializes a new VehicleDTO instance.

        Args:
            license_plate (str): License plate number.
            brand (str): Brand of the vehicle.
            apk_expiry_date (str): Expiry date of the APK (Algemene Periodieke Keuring) inspection.
            veh_registration_nr (str): Type approval number of the vehicle registration.
        """
    def __init__(self, license_plate, brand, apk_expiry_date, veh_registration_nr):
        self.license_plate = license_plate
        self.brand = brand
        self.apk_expiry_date = apk_expiry_date
        self.veh_registration_nr = veh_registration_nr

    """
        Creates a VehicleDTO object from JSON data.

        Args:
            data (dict): JSON data containing vehicle information.

        Returns:
            VehicleDTO: A VehicleDTO object with information extracted from the JSON data.
        """
    @staticmethod
    def from_json(data):
        # Access specific values using the keys
        license_plate = data[0]['kenteken']
        brand = data[0]['merk']
        apk_expiry_date = data[0]['vervaldatum_apk_dt']
        veh_registration_nr = data[0]['typegoedkeuringsnummer']   
        return VehicleDTO(license_plate, brand, apk_expiry_date, veh_registration_nr)