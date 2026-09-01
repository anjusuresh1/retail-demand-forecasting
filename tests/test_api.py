def test_openapi_contains_expected_routes():
    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/health" in paths
    assert "/v1/predict/demand" in paths
    assert "/v1/recommend/price" in paths