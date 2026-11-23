import yaml
from telepath_worker import storage_layer

TELEPATH_SPEC_VERSION = 2

class SpecError(Exception):
    """Exception raised for errors in the spec.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, spec_path, message):
        self.spec_path = spec_path
        self.message = message


class Spec(object):
    spec = {}
    required = None
    spec_label = None

    def __init__(self):
        self.required = ['telepath_spec',]   # spec keys that are required

    def check_required(self,path=None):
        for r in self.required:
            if r not in self.spec:
                raise SpecError( path, r +" : key not found")

    def save(self, path):

        # TODO: Ensure ALL specs are within minimal wrapper,
        # make writing much easier.
        # ReplaySpec is the one that isnt...

        # this function not proven yet...

        fullspec = {}
        fullspec['telepath_spec']=TELEPATH_SPEC_VERSION
        fullspec['spec_type']=self.spec_label

        yamls = yaml.dump_all( [self.spec,] )
        print("yamls",yamls)
        if storage_layer.is_s3_location(path):
            bucket, key = storage_layer.parse_s3_path(path)
            storage_layer.write_s3(yamls, bucket, key)
        else:
            with open(path,'w') as f:
                documents = yaml.dump_all( [self.spec,], f)

    def _load(self, path):
        if storage_layer.is_s3_location(path):
            s=storage_layer.read_path_from_s3(path)
            self.spec = yaml.load(s)
        else:
            with open(path) as f:
                self.spec= yaml.load(f, Loader=yaml.FullLoader)
                self.check_required( path )
        return self.spec

    def load(self, path):
        self._load(path)
        if self.spec['spec_type'] != self.spec_label:
            raise SpecError( path,"not a spec type "+self.spec_label)
        return self.spec



class SettingsSpec( Spec ):
    spec_label = 'settings'

    def __init__(self):
        super().__init__()



class AgentSpec( Spec ):
    spec_label = 'agent'

    def __init__(self):
        super().__init__()
        self.required.append('spec_subtype')


class EnvSpec( Spec ):
    spec_label = 'env'

    def __init__(self):
        super().__init__()
        self.required.append('env_label')


class ReplayPrepSpec( Spec ):
    spec_label = 'replay_prep'

    def __init__(self):
        super().__init__()


class ReplaySpec( Spec ):
    spec_label = 'replay'

    def __init__(self):
        super().__init__()



class AlgoSpec( Spec ):
    spec_label = 'algo'

    def __init__(self):
        super().__init__()

