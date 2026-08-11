# Formal authoring runtime

This deployment uses one immutable image for the authoring worker and the
HTTPS CONNECT proxy. The authoring job joins only a Docker `--internal`
network. The proxy is the only dual-homed peer and permits exactly the locked
provider hostname on TCP 443; IP literals, other hostnames, and other ports are
rejected.

`deploy.ps1` builds the image from a digest-pinned Python base, creates the
networks and proxy, runs positive and negative probes, and publishes one
canonical lock bundle. The lock directory is mandatory and must be a new
`deploy/authoring/locks-vN` namespace, for example
`--runtime-version formal-v3 --lock-directory deploy/authoring/locks-v4`.
The script refuses an existing lock directory and publishes the four files
through a staging directory, so historical lock bytes cannot be overwritten.
If image inputs change after deployment, use both a new runtime version and a
new lock namespace; never reuse the previous namespace. Runtime versions also
create separately named image, network, and proxy resources.

The repository-level `.dockerignore` is an allowlist: only `src/`, this
Dockerfile, and the hash-locked requirements file enter the build context.
Local `.env`, datasets, run artifacts, Git metadata, and unrelated workspace
files are never sent to the build daemon.

The generated sandbox profile is host-specific because it commits the Docker
CLI path/hash, Docker network ID, image ID, policy digest, and proxy identity.
It contains no credentials. A formal authoring call still requires the
provider API key in the parent process and passes it by environment-variable
name only.
