
# RESULT

- BatchResult
    - collect success and errors

- Errors
    - to include the error information or even the error object

- Abort mission
    - when we accumulate enough errors -> we cancel the execution
    - therefore we must track some "total" state

- We have a smart way of controlling the speed at which we run the evaluations
  ie. fill the 'executor'
  - To control using the primitive from asyncio, the 'semaphore'

- To check the webhook proxy address that it's resolvable
    -> by requesting the token from the proxy that we will use for listening
    -> the make_request function does not check for the webhook proxy to be resolvable
        -> therefore the control flow must exist so that it's checked pre-execution

- Setting the retry parameters - Where should it optimally happen?


- We don't have the Baseten and Proxy error possibilities / schema, we need to get them

- We don't properly resolve the webhook proxy and we send requests to Baseten incurring costs
without being certain they will be able to send the results to the webhook ever

- The token for the being allowed to listen and open the stream, should be requested
before the generation, and hence we should create the token before the generation on the client,
that way we don't need to pass the request id to the proxy

- https://docs.baseten.co/api-reference/get-async-request-status




# TODOS

1. Figure if we want to have the parser be passed the fail_on_parse_error flag
   flow_judge - AsyncFlowJudge - async_batch_evaluate
