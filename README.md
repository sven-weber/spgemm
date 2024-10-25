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

## Run on euler

TODO!!

```console
source init-euler.sh
```

```console
pip3 install -r requirements.txt
```