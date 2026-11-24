/* nvmtl_schedule_policy.h - how much of a command buffer's surface writes are published before it reports "scheduled".
   Modes: 3 two-plane surfaces written through private images, 2 every private image, 1 every surface write, 0 none.
   Pure, so tools/test_schedule_policy.c proves it on the host. */
#ifndef NVMTL_SCHEDULE_POLICY_H
#define NVMTL_SCHEDULE_POLICY_H
#include <string.h>

/* The Chromium family hands layers to another process from its scheduled callbacks, and that process has no queue
   dependency on ours, so these processes publish every surface write first. */
static inline int nvmtl_schedule_is_chromium_family(const char *process)
{
    if (!process) return 0;
    return strstr(process, "Google Chrome") || strstr(process, "Chromium") || strstr(process, "Microsoft Edge");
}
/* `setting` is NVMTL_SCHEDULE_AFTER_SURFACE_WRITES (may be NULL). Only a single digit 0..3 is an override; anything else
   is ignored rather than turned into a mode number that no code path knows. */
static inline int nvmtl_schedule_after_mode(const char *setting, const char *process)
{
    if (setting && setting[0] >= '0' && setting[0] <= '3' && !setting[1]) return setting[0] - '0';
    return nvmtl_schedule_is_chromium_family(process) ? 1 : 3;
}

/* Browser and Electron GPU helper processes hand layers across processes from scheduled callbacks without NVMTLBusyAt
   ordering, so they keep mode 1 (publish every surface write before reporting "scheduled"). */
static inline int nvmtl_schedule_is_ui_browser_family(const char *process)
{
    if (!process || !*process) return 0;
    if (nvmtl_schedule_is_chromium_family(process)) return 1;
    return strstr(process, "Helper (GPU)") || strstr(process, "Electron")
        || strstr(process, "Brave") || strstr(process, "Opera")
        || strstr(process, "Arc") || strstr(process, "Vivaldi");
}

/* Mode 3 for WindowServer, mode 1 for Chromium/Electron GPU helpers (which rely on scheduledHandlers for cross-process
   layer handoff), and mode 2 for native macOS apps (App Store, NotificationCenter, WidgetRenderer, Safari, Finder):
   private/non-VRAM copy-out surfaces still publish before "scheduled", while shared family-VRAM surfaces pipeline and
   synchronize via NVMTLBusyAt + nvmtlSurfaceIn. NVMTL_SCHEDULE_AFTER_SURFACE_WRITES=0..3 still overrides everywhere. */
static inline int nvmtl_schedule_after_mode_ui(const char *setting, const char *process)
{
    if (setting && setting[0] >= '0' && setting[0] <= '3' && !setting[1]) return setting[0] - '0';
    if (!process || !*process || !strcmp(process, "WindowServer")) return 3;
    if (nvmtl_schedule_is_ui_browser_family(process)) return 1;
    return 2;
}
#endif
