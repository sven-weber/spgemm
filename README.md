# spgemm
SpGEMM implementation for DPHPC at ETH


## Install on MacOS

To tun the code on MacOS, we need to override the used compiler.
For this we will use the [direnv]() tool that will automatically do this for us every time we open this repository.

Follow the following steps:

1. Hook direnv to your shell. Append the following line to your `~/.zshrc` file:

```console
eval "$(direnv hook zsh)"
```

2. Execute the install script.

This repository has a script to install all dependencies and perform the setup for your. Execute:

```console
./scripts/install_deps_macos.sh
```

## Fetch matrices

```console
python3 scripts/fetch_matrices.py
```

## Run on euler

```console
source euler/init.sh
```

```console
./euler/get_euler_nodes.sh
```

```console
pip3 install -r requirements.txt
```

```console
python3 scripts/fetch_matrices.py --euler
```

Build for release:

```console
make optimize
```

Run the benchmark:

```console
python3 scripts/sweep_benchmark.py --impl baseline --matrix first --min 10 --max 10 --stride 1 --euler
```

## Run on Piz Daint (CSCS)

1. Generate key-pair via:

[https://sshservice.cscs.ch/](https://sshservice.cscs.ch/)

2. Copy those key-pairs into your ssh folder

3. Run

```console
ssh-add -t 1d ~/.ssh/cscs-key
```

4. Add the following to your ssh config

```bash
Host daint
    HostName daint
    User ltagliav
    ProxyJump ltagliav@ela.cscs.ch
    AddKeysToAgent yes
    IdentityFile ~/.ssh/cscs-key
    ForwardAgent yes
```

5. Connect via

```console
ssh daint
```