/** @type {import('next').NextConfig} */
const nextConfig = {
  // A static site: no server, no API routes, no server-side numerical logic. The
  // build emits plain HTML/CSS/JS into out/, which Netlify serves as files.
  output: "export",
  reactStrictMode: true,
  trailingSlash: true,
  // Static export cannot use the Image Optimization server, and the figures are
  // pre-rendered PNGs from the C++ artifacts, so they are served as-is.
  images: { unoptimized: true },
};

export default nextConfig;
