from instrument.instrument import Instrument


def test_load_json():
    d = Instrument.load_parameter()
    print(d)
