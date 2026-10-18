import type { Metadata } from "next";
import "./styles.css";

export const metadata: Metadata = {
  title: "Fieldnote — product discovery",
  description: "Find products by describing what you need and selecting examples.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
