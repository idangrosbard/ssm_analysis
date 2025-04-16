from abc import ABCMeta


class HMRCompatibleMeta(ABCMeta):
    def __instancecheck__(cls, instance):
        return (
            isinstance(type(instance), type)
            and type(instance).__name__ == cls.__name__
            and type(instance).__module__ == cls.__module__
        )
