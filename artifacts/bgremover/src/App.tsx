import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import NotFound from '@/pages/not-found';
import { Route, Switch, Router as WouterRouter } from 'wouter';
import { Layout } from '@/components/layout/Layout';
import { Home } from '@/pages/home';
import { BackgroundRemoval } from '@/pages/background-removal';
import { Batch } from '@/pages/batch';
import { History } from '@/pages/history';
import { Admin } from '@/pages/admin';
import { Generate } from '@/pages/generate';

const queryClient = new QueryClient();

function Router() {
  return (
    <Switch>
      <Route path="/" component={Home} />
      <Route path="/generate" component={Generate} />
      <Route path="/remove-background" component={BackgroundRemoval} />
      <Route path="/batch" component={Batch} />
      <Route path="/history" component={History} />
      <Route path="/admin" component={Admin} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={150}>
        <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}>
          <Layout>
            <Router />
          </Layout>
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
