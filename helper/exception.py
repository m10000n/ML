# this file is used before the python environment is ready

import textwrap


class ValidationError(Exception):
    def __init__(self, param_name: str, constraint: str, value: str):
        message = textwrap.dedent(
            f"""
            Validation of parameter `{param_name}` failed.
                Constraint: {constraint}
                Value: {value}
            """
        )

        super().__init__(message)
        self.param_name = param_name
        self.constraint = constraint
        self.value = value


class PreconditionError(Exception):
    def __init__(self, expected: str, actual: str):
        message = textwrap.dedent(
            f"""
            Precondition violated.
                Expected: {expected}
                Actual: {actual}
            """
        )

        super().__init__(message)
        self.expected = expected
        self.actual = actual


class NanError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class DataNotFoundError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
