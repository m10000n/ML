from data.data_loader import DataLoadersConfig


def batch_size_n(n: int) -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=n, shuffle_first=True)


def batch_size_1() -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=1, shuffle_first=True)


def batch_size_2() -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=2, shuffle_first=True)


def batch_size_4() -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=4, shuffle_first=True)


def batch_size_8() -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=8, shuffle_first=True)


def batch_size_16() -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=16, shuffle_first=True)


def batch_size_32() -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=32, shuffle_first=True)


def batch_size_64() -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=64, shuffle_first=True)


def batch_size_128() -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=128, shuffle_first=True)


def batch_size_256() -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=256, shuffle_first=True)


def batch_size_512() -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=512, shuffle_first=True)


def batch_size_1024() -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=1024, shuffle_first=True)


def batch_size_2048() -> DataLoadersConfig:
    return DataLoadersConfig(batch_size=2048, shuffle_first=True)
