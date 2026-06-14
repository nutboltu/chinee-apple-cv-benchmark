# Connect VS Code to a Colab GPU runtime

One-time setup, then run the bootstrap notebook each Colab session.

## 1. One-time on your Mac

```bash
brew install cloudflared
code --install-extension ms-vscode-remote.remote-ssh
```

## 2. Each Colab session

1. Open <https://colab.research.google.com>, upload `colab_ssh_bootstrap.ipynb` (or `File → Open notebook → GitHub`).
2. **Runtime → Change runtime type → GPU**.
3. Run all cells. Enter a password when prompted.
4. The last cell prints something like:

   ```
   HostName: <words>-<words>.trycloudflare.com
   User: root
   ```

5. Append (or replace the existing `Host colab` block) in `~/.ssh/config`:

   ```sshconfig
   Host colab
       HostName <words>-<words>.trycloudflare.com
       User root
       ProxyCommand cloudflared access ssh --hostname %h
       StrictHostKeyChecking no
       UserKnownHostsFile /dev/null
   ```

6. In VS Code: `Cmd+Shift+P → Remote-SSH: Connect to Host → colab`. Enter the password from step 3.

The Colab VM workspace lives at `/content`. Clone the repo there (uncomment the last notebook cell) and you've got a normal VS Code dev loop on Colab's GPU.

## Caveats

- The tunnel URL changes every session — update `~/.ssh/config` each time.
- Colab kills idle VMs at ~90 min and caps sessions at 12 h.
- Closing the Colab browser tab terminates the VM.
