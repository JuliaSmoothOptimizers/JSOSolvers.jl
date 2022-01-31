using Pkg
Pkg.add("GitHub")
Pkg.instantiate()

using GitHub

ORG, REPO, PR = ARGS[1], ARGS[2], ARGS[3]
TEST_RESULTS_FILE = "$(ORG)_$(REPO)_$(PR).txt"

# Need to add GITHUB_AUTH to your .bashrc
myauth = GitHub.authenticate(ENV["GITHUB_AUTH"])

function create_gist(authentication)

    file_content = read(TEST_RESULTS_FILE, String)
    file_dict = Dict(TEST_RESULTS_FILE => Dict("content" => file_content))
    gist = Dict{String,Any}(
        "description" => "Tuning results:",
        "public" => true,
        "files" => file_dict,
    )

    posted_gist = GitHub.create_gist(params = gist, auth = authentication)

    return posted_gist
end

function post_gist_url_to_pr(comment::String; kwargs...)
    api = GitHub.DEFAULT_API
    repo = get_repo(api, ORG, REPO; kwargs...)
    pull_request = get_pull_request(api, ORG, repo, parse(Int, PR); kwargs...)
    GitHub.create_comment(api, repo, pull_request, comment; kwargs...)
end

function get_repo(api::GitHub.GitHubWebAPI, org::String, repo_name::String; kwargs...)
    my_params = Dict(:visibility => "all")
    # return GitHub.repo(api, repo; params = my_params, kwargs...)
    return Repo(
        GitHub.gh_get_json(
            api,
            "/repos/$(org)/$(repo_name)";
            params = my_params,
            kwargs...,
        ),
    )
end

function get_pull_request(
    api::GitHub.GitHubWebAPI,
    org::String,
    repo::Repo,
    pullrequest_id;
    kwargs...,
)
    my_params = Dict(:sort => "popularity", :direction => "desc")
    pull_request = PullRequest(
        GitHub.gh_get_json(
            api,
            "/repos/$(org)/$(repo.name)/pulls/$(pullrequest_id)";
            params = my_params,
            kwargs...,
        ),
    )

    return pull_request
end

function get_comment_from_test_results()
    open(TEST_RESULTS_FILE, "r") do file
        text_to_match = r"Best feasible solution"
        for line in readlines(file)
            if occursin(text_to_match, line)
                return "$(strip(line)): "
            end
        end
        return "Tuning failed: "
    end
end

comment = get_comment_from_test_results()
post_gist_url_to_pr("$comment $(create_gist(myauth).html_url)"; auth = myauth)
