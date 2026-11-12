// Hand-drawn 20px line icons, so no icon library is shipped.
interface IconProps {
  size?: number;
}

const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: "0 0 20 20",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
});

export const IconOverview = ({ size = 20 }: IconProps) => (
  <svg {...base(size)}>
    <rect x="3" y="3" width="6" height="6" rx="1.5" />
    <rect x="11" y="3" width="6" height="4" rx="1.5" />
    <rect x="11" y="9" width="6" height="8" rx="1.5" />
    <rect x="3" y="11" width="6" height="6" rx="1.5" />
  </svg>
);

export const IconPulse = ({ size = 20 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M2 11h3.5l2-5 3 9 2.5-6 1.5 2H18" />
  </svg>
);

export const IconList = ({ size = 20 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M7 5h10M7 10h10M7 15h10" />
    <circle cx="3.5" cy="5" r="0.9" />
    <circle cx="3.5" cy="10" r="0.9" />
    <circle cx="3.5" cy="15" r="0.9" />
  </svg>
);

export const IconSplit = ({ size = 20 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M3 10h5M8 10l4-5h5M8 10l4 5h5" />
  </svg>
);

export const IconShield = ({ size = 20 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M10 2.5 16 5v4.5c0 3.8-2.6 6.6-6 8-3.4-1.4-6-4.2-6-8V5z" />
    <path d="m7.2 10 2 2 3.8-4" />
  </svg>
);

export const IconBook = ({ size = 20 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M3 4.5c2.5-1 5-1 7 .5 2-1.5 4.5-1.5 7-.5V16c-2.5-1-5-1-7 .5-2-1.5-4.5-1.5-7-.5z" />
    <path d="M10 5v11.5" />
  </svg>
);

export const IconPlay = ({ size = 18 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M6 4.5v11l9-5.5z" fill="currentColor" stroke="none" />
  </svg>
);

export const IconPause = ({ size = 18 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M7 4.5v11M13 4.5v11" strokeWidth={2.4} />
  </svg>
);
