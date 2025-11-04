import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  turbopack: {
    // Explicitly set Turbopack root to this app to avoid monorepo lockfile inference warnings
    root: __dirname,
  },
};

export default nextConfig;
