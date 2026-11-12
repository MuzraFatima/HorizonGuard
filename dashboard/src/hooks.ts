import { useEffect, useState } from "react";
import { get, getCached } from "./api";

interface ApiState<T> {
  data: T | undefined;
  error: string | null;
  loading: boolean;
}

/** Fetch a GET endpoint through the shared cache; re-runs when `path` changes. */
export function useApi<T>(path: string | null): ApiState<T> {
  const [state, setState] = useState<ApiState<T>>(() => {
    const cached = path ? getCached<T>(path) : undefined;
    return { data: cached, error: null, loading: path !== null && cached === undefined };
  });

  useEffect(() => {
    if (path === null) return;
    let alive = true;
    const cached = getCached<T>(path);
    if (cached !== undefined) {
      setState({ data: cached, error: null, loading: false });
      return;
    }
    setState((s) => ({ ...s, loading: true, error: null }));
    get<T>(path)
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((e: Error) => alive && setState({ data: undefined, error: e.message, loading: false }));
    return () => {
      alive = false;
    };
  }, [path]);

  return state;
}

/** Debounce a fast-changing value, such as a search box. */
export function useDebounced<T>(value: T, delay = 300): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(id);
  }, [value, delay]);
  return debounced;
}
