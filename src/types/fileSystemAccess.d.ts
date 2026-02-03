export {};

declare global {
  interface FileSystemHandlePermissionDescriptor {
    mode?: 'read' | 'readwrite';
  }

  interface FileSystemHandle {
    requestPermission?(
      descriptor?: FileSystemHandlePermissionDescriptor,
    ): Promise<PermissionState>;
    queryPermission?(
      descriptor?: FileSystemHandlePermissionDescriptor,
    ): Promise<PermissionState>;
  }
}
