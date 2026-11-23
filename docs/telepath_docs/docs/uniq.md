## Unique Agent Names

If you run the same training command line twice, with the same agent output file name, they will clobber each other.  You may not want that.

Telepath can provide unique agent filenames.  In your agent path, include the string `AGENTID`.  It will be replaced with a random hash.  

For example, using the following cmd line...:

```
telepath train ... --agent_path=s3://bucketname/folder1/agent_batch_AGENTID.agent_telepath ...
```

... Telepath will construct a unique agent path like this, for each execution:
```
s3://bucketname/folder1/agent_batch_A14a9b2034dC4.agent_telepath
```

Unique agent paths means you can safely run the exact same training command line in multiple times without clobbering the resulting agent files.  Both one after the other, or at the same time with [procx](procx.md).
