# Register reduceUSANS shell completion when activating the Pixi/conda environment.
#
# The reduceUSANS command is generated from the pyproject.toml console-script
# entry point, so users do not invoke src/usansred/reduce.py directly. Register
# the generated command name here to make tab completion work after `pixi shell`
# or conda-style environment activation.

case "$-" in
    *i*) ;;
    *) return 0 2>/dev/null || exit 0 ;;
esac

if [ "${USANSRED_ARGCOMPLETE_REGISTERED:-}" = "1" ]; then
    return 0 2>/dev/null || exit 0
fi

if ! command -v register-python-argcomplete >/dev/null 2>&1; then
    return 0 2>/dev/null || exit 0
fi

if [ -n "${BASH_VERSION:-}" ]; then
    eval "$(register-python-argcomplete --shell bash reduceUSANS 2>/dev/null)"
    export USANSRED_ARGCOMPLETE_REGISTERED=1
elif [ -n "${ZSH_VERSION:-}" ]; then
    autoload -Uz compinit 2>/dev/null || true
    command -v compdef >/dev/null 2>&1 || compinit -i 2>/dev/null || return 0
    eval "$(register-python-argcomplete --shell zsh reduceUSANS 2>/dev/null)"
    export USANSRED_ARGCOMPLETE_REGISTERED=1
fi
