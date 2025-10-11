


export function createPageUrl(pageName: string) {
  return (
    '/' +
    pageName
      .trim()
      // insert hyphen between lowercase/number and uppercase boundary
      .replace(/([a-z0-9])([A-Z])/g, '$1-$2')
      // normalize spaces/underscores to hyphen
      .replace(/[\s_]+/g, '-')
      .toLowerCase()
  );
}