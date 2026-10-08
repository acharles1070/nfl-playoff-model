#!/bin/zsh
# Install (or remove) the weekly LaunchAgent.   ./scripts/install_scheduler.sh [install|remove|status]
LABEL="com.nfl-playoff-model.weekly"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
SCRIPT="${NFL_REPO:-$HOME/Developer/nfl-playoff-model}/scripts/weekly_log.sh"
DOMAIN="gui/$(id -u)"

case "${1:-install}" in
  install)
    mkdir -p "$HOME/Library/LaunchAgents" "$HOME/Library/Logs/nfl-playoff-model"
    cat > "$PLIST" <<PLISTEOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>$SCRIPT</string></array>
  <key>StartCalendarInterval</key><array>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>12</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>17</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$HOME/Library/Logs/nfl-playoff-model/launchd.out</string>
  <key>StandardErrorPath</key><string>$HOME/Library/Logs/nfl-playoff-model/launchd.err</string>
  <key>RunAtLoad</key><false/>
</dict></plist>
PLISTEOF
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null
    launchctl bootstrap "$DOMAIN" "$PLIST" && echo "installed: $PLIST" ;;
  remove)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null; rm -f "$PLIST"; echo "removed" ;;
  status)
    launchctl print "$DOMAIN/$LABEL" 2>&1 | grep -E "state|path|weekday|hour|minute|runs|last exit" | head -20 ;;
esac
